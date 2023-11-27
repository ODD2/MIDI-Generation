# for eval_metrics.py
`git clone https://github.com/slSeanWU/MusDr.git`
you need to move eval_metrics.py in the MusDr
`python eval_metrics.py --dict_path [the dictionary path] --output_file_path [the output file path]`

if you want to use your own dictionary please make sure to specify the BAR_EV
POS_EVS and PITCH_EVS must be sorted.

eg 
POS_EVS in basic_event_dictionary.pkl are 1~17 represent Position_1/16~Position_16/16
PITCH_EVS in basic_event_dictionary.pkl are 99~185 represent Note On_22~Note On_107