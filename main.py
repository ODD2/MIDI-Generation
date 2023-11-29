import os
import glob
import wandb
import utils
import torch
import pickle
import random
import argparse
import warnings
import pandas as pd
import numpy as np


from torch import nn
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from transformers import TransfoXLConfig, TransfoXLModel
from torch.utils.data.dataloader import DataLoader, Dataset
from midi2audio import FluidSynth
from pydub import AudioSegment


warnings.filterwarnings('ignore')


# set the input length. must be same with the model config
X_LEN = 1024


def parse_opt():
    parser = argparse.ArgumentParser()
    ####################################################
    # you can define your arguments here. there is a example below.
    # parser.add_argument('--device', type=str, help='gpu device.', default='cuda')
    ####################################################
    parser.add_argument('--dict_path', type=str, help='the dictionary path.', default='./basic_event_dictionary.pkl')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ckp_folder', type=str, default='./checkpoints/')
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--group_size', type=int, default=2)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--ckp_path', type=str, default="")
    parser.add_argument('--out_prefix', type=str, default="")
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--model_ver', type=str, default="v1")
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1.5)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--nucleus_p', type=float, default=0.9)
    parser.add_argument('--sample_mode', type=str, default="topk")

    args = parser.parse_args()
    return args


opt = parse_opt()
event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))


def set_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


set_seed(1019)


class NewsDataset(Dataset):
    def __init__(self, midi_l=[]):
        if (opt.debug):
            midi_l = midi_l[:30]
        else:
            midi_l = midi_l[:int(len(midi_l) * opt.ratio)]
        self.midi_l = midi_l
        self.x_len = X_LEN
        self.dictionary_path = opt.dict_path
        self.event2word, self.word2event = pickle.load(open(self.dictionary_path, 'rb'))
        self.parser = self.prepare_data(self.midi_l)

    def __len__(self):
        return len(self.parser)

    def __getitem__(self, index):
        return self.parser[index]

    def chord_extract(self, midi_path, max_time):
        ####################################################
        # add your chord extracttion method here if you want
        ####################################################
        return

    def extract_events(self, input_path):
        event_path = input_path.replace(".mid", ".events")
        if (os.path.exists(event_path)):
            with open(event_path, "rb") as f:
                events = pickle.load(f)
        else:
            note_items, tempo_items = utils.read_items(input_path)
            note_items = utils.quantize_items(note_items)
            max_time = note_items[-1].end

            # if you use chord items you need add chord_items into "items"
            # e.g : items = tempo_items + note_items + chord_items
            items = tempo_items + note_items

            groups = utils.group_items(items, max_time)
            events = utils.item2event(groups)

            with open(event_path, "wb") as f:
                pickle.dump(events, f)

        return events

    def prepare_data(self, midi_paths):
        # extract events
        all_events = []
        for path in tqdm(midi_paths):
            events = self.extract_events(path)
            all_events.append(events)
        # event to word
        all_words = []
        for events in tqdm(all_events):
            words = []
            for event in events:
                e = '{}_{}'.format(event.name, event.value)
                if e in self.event2word:
                    words.append(self.event2word[e])
                else:
                    # OOV
                    if event.name == 'Note Velocity':
                        # replace with max velocity based on our training data
                        words.append(self.event2word['Note Velocity_21'])
                    else:
                        # something is wrong
                        # you should handle it for your own purpose
                        print('something is wrong! {}'.format(e))
            all_words.append(words)

        # all_words is a list containing words list of all midi files
        # all_words = [[tokens of midi], [tokens of midi], ...]

        # you can cut the data into what you want to feed to model
        # Warning : this example cannot use in transformer_XL you must implement group segments by yourself
        segments = []
        for words in all_words:
            pairs = []
            for i in range(0, len(words) - self.x_len - 1, self.x_len):
                x = words[i:i + self.x_len]
                y = words[i + 1:i + self.x_len + 1]
                pairs.append([x, y])
            # abandon last segments in a midi
            # pairs = pairs[0:len(pairs) - (len(pairs) % opt.group_size)]
            if (len(pairs) >= opt.group_size):
                segments.append(pairs[:opt.group_size])
        segments = np.array(segments)
        print(segments.shape)
        return segments


class Model(nn.Module):
    def __init__(self, words=249):
        super(Model, self).__init__()
        #################################################
        # create your model here
        #################################################
        configuration = TransfoXLConfig()
        if (opt.model_ver == "v1"):
            configuration.d_embed = 768
            configuration.d_model = 768
            configuration.d_inner = 768 * 4
            configuration.d_head = 64
            configuration.n_head = 12
            configuration.n_layer = 12
            configuration.mem_len = X_LEN
        elif (opt.model_ver == "v2"):
            configuration.n_layer = 6
            configuration.mem_len = X_LEN
        self.model = TransfoXLModel(configuration)
        self.linear = nn.Sequential(
            nn.LayerNorm(configuration.d_embed),
            nn.Dropout(),
            nn.Linear(configuration.d_embed, words),
        )

    def forward(self, x, mems=None):
        #################################################
        # create your model here
        #################################################
        y = self.model(x, mems=mems)
        logits = self.linear(y['last_hidden_state'])
        mems = y["mems"]
        return dict(
            logits=logits,
            mems=mems
        )


def temperature_sampling(
        logits,
        temperature,
        mode,
        topk,
        nucleus_p
):
    #################################################
    # 1. adjust softmax with the temperature parameter
    # 2. choose topk highest probs
    # 3. normalize the topk highest probs
    # 4. random choose one from the topk highest probs as result by the probs after normalize
    #################################################
    smax_temper = (logits / temperature).softmax(dim=-1)
    values, indices = smax_temper.sort(descending=True)
    if mode == "topk":
        pass
    elif mode == "nucleus":
        cur_p = 0
        for top_k in range(indices.shape[0]):
            cur_p += values[top_k]
            if (cur_p >= nucleus_p):
                break
    else:
        raise NotImplementedError()

    indices = indices[:topk].numpy()
    values = values[:topk].numpy()
    word = np.random.choice(
        indices,
        p=values / values.sum()
    )
    return word


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


@torch.inference_mode()
def test(
    n_target_bar=32,
    temperature=1.5,
    topk=5,
    sample_mode="topk",
    nucleus_p=0.95,
    model_path='',
    num_samples=20,
    out_prefix="",
):
    path_split = model_path.split('/')
    path_split[0] = "results"
    path_split[-1] = path_split[-1].replace(".pkl", "")
    if (len(out_prefix) > 0):
        path_split.insert(2, out_prefix)
    out_folder = '/'.join(path_split)
    os.makedirs(out_folder, exist_ok=True)

    event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))
    fs = FluidSynth()

    with torch.no_grad():
        # load model
        checkpoint = torch.load(model_path)
        model = Model().to(opt.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        batch_size = 1
        for sample_i in range(num_samples):
            out_midi_path = os.path.join(out_folder, f"{sample_i}.midi")
            words = []
            for _ in range(batch_size):
                ws = [event2word['Bar_None']]
                if 'chord' in opt.dict_path:
                    tempo_classes = [v for k, v in event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in event2word.items() if 'Tempo Value' in k]
                    chords = [v for k, v in event2word.items() if 'Chord' in k]
                    ws.append(event2word['Position_1/16'])
                    ws.append(np.random.choice(chords))
                    ws.append(event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                else:
                    tempo_classes = [v for k, v in event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in event2word.items() if 'Tempo Value' in k]
                    ws.append(event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                words.append(ws)

            # generate
            original_length = len(words[0])
            initial_flag = 1
            current_generated_bar = 0
            mems = None
            print('Start generating')
            while current_generated_bar < n_target_bar:
                print("\r", current_generated_bar, end="")
                # input
                if initial_flag:
                    temp_x = np.zeros((batch_size, original_length))
                    for b in range(batch_size):
                        for z, t in enumerate(words[b]):
                            temp_x[b][z] = t
                    initial_flag = 0
                else:
                    temp_x = np.zeros((batch_size, 1))
                    for b in range(batch_size):
                        temp_x[b][0] = words[b][-1]

                temp_x = torch.Tensor(temp_x).long()

                output = model(temp_x.to(opt.device), mems=mems)
                logits = output["logits"]
                mems = output["mems"]

                # sampling
                _logit = logits[0, -1].to('cpu').detach()
                word = temperature_sampling(
                    logits=_logit,
                    temperature=temperature,
                    mode=sample_mode,
                    topk=topk,
                    nucleus_p=nucleus_p
                )

                words[0].append(word)

                if word == event2word['Bar_None']:
                    current_generated_bar += 1

            utils.write_midi(
                words=words[0],
                word2event=word2event,
                output_path=out_midi_path,
                prompt_path=None
            )

            out_audio_path = out_midi_path.replace(".midi", ".wav")
            fs.midi_to_audio(out_midi_path, out_audio_path)

            sound = AudioSegment.from_file(out_audio_path, format="wav")
            normalized_sound = match_target_amplitude(sound, -20.0)
            normalized_sound.export(out_audio_path, format="wav")


def train(checkpoints_path=''):
    epochs = 200

    # create data list
    # use glob to get all midi file path
    train_list = glob.glob('dataset/midi_analyzed/*/*.mid')
    print('train list len =', len(train_list))

    # dataset
    train_dataset = NewsDataset(train_list)

    # dataloader
    assert opt.batch_size % opt.group_size == 0
    opt.batch_size = int(opt.batch_size // opt.group_size)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    print('Dataloader is created')

    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device(opt.device)
    else:
        device = torch.device("cpu")

    # create model
    start_epoch = 1
    global_steps = 0
    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scaler = GradScaler()

    if (len(checkpoints_path) > 0):
        # wheather checkpoint_path is exist
        if os.path.isfile(checkpoints_path):
            checkpoint = torch.load(checkpoints_path)
        else:
            os._exit()
        start_epoch = checkpoint['epoch'] + 1
        global_steps = checkpoint["global_steps"] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint["scaler"])

    # init wandb
    wandb.init(
        project="NTUHW3",
        mode=("offline"if opt.debug else "online")
    )
    wandb.watch(models=model, log="gradients", log_freq=100)

    print('Model is created \nStart training')
    model.train()
    losses = []

    opt.ckp_folder = os.path.join(opt.ckp_folder, wandb.run.id)

    try:
        os.makedirs(opt.ckp_folder, exist_ok=True)
        print("Checkpoint folder created.")
    except:
        pass

    for epoch in range(start_epoch, epochs + 1):
        wandb.log(
            {
                "epoch": epoch,
                "lr": get_lr(optimizer)
            },
            step=global_steps
        )
        single_epoch = []

        for data in tqdm(train_dataloader):
            loss = 0
            mems = None
            data = data.long().to(device)
            for j in range(opt.group_size):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # x, y shape = (batch_size, length)
                    x = data[:, j, 0, :]
                    y = data[:, j, 1, :]
                    output = model(x, mems=mems)
                    logits = output["logits"]
                    mems = output["mems"]
                    _loss = nn.functional.cross_entropy(logits.permute(0, 2, 1), y)
                scaler.scale(_loss).backward()
                loss += _loss.detach().cpu().item() / opt.group_size
            single_epoch.append(loss)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_steps += 1
            wandb.log(
                {"train/loss": loss},
                step=global_steps
            )

        single_epoch = np.array(single_epoch)
        losses.append(single_epoch.mean())
        print('>>> Epoch: {}, Loss: {:.5f}'.format(epoch, losses[-1]))
        torch.save(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'loss': losses[-1],
                'global_steps': global_steps
            },
            os.path.join(opt.ckp_folder, 'epoch_%03d.pkl' % epoch)
        )
        np.save(os.path.join(opt.ckp_folder, 'training_loss'), np.array(losses))

    wandb.finish()


def main():
    ######################################
    # write your main function here
    ######################################
    if (opt.mode == "train"):
        train(checkpoints_path=opt.ckp_path)
    elif (opt.mode == "test"):
        test(
            model_path=opt.ckp_path,
            num_samples=opt.num_samples,
            topk=opt.topk,
            sample_mode=opt.sample_mode,
            nucleus_p=opt.nucleus_p,
            temperature=opt.temperature,
            out_prefix=opt.out_prefix
        )
    else:
        raise NotImplementedError()
    return


if __name__ == '__main__':
    main()
