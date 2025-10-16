# FCIAC
 Few-shot Class-adjustable Incremental Audio Classification(FCIAC), which can  handle dynamic scenarios where classes can be both added and removed under few-shot conditions.

## Datasets
To study the Few-shot Class-incremental Audio Classification (FCAC) problem, three datasets of LS-100 dataset, NSynth-100 dataset and FSC-89 dataset are constructed by 
choosing samples from audio corpora of the [Librispeech](https://www.openslr.org/12/) dataset, the [NSynth](https://magenta.tensorflow.org/datasets/nsynth) dataset and the [FSD-MIX-CLIPS](https://zenodo.org/record/5574135#.YWyINEbMIWo) dataset respectively.
In the study of the FCIAC problem, we still use these three datasets.

Wei Xie, one of our team members, constructed the NSynth-100 dataset and FSC-89 dataset. The detailed information of these two datasets is [here](https://github.com/chester-w-xie/FCAC_datasets).

The detailed information of the LS-100 dataset is given below.

### Statistics on the LS-100 dataset

|                                                                 | LS-100                                        |
|:---------------------------------------------------------------:|:---------------------------------------------:|
| Type of audio                                                   | Speech                                        |
| Num. of classes                                                 | 100 (60 of base classes, 40 of novel classes) |
| Num. of training / validation / testing samples per base class  | 500 / 150 / 100                               |
| Num. of training / validation / testing samples per novel class | 500 / 150 / 100                               |
| Duration of the sample                                          | All in 2 seconds                              |
| Sampling frequency                                              | All in 16K Hz                                 |

### Preparation of the LS-100 dataset

LibriSpeech is a corpus of approximately 1000 hours of 16kHz read English speech, prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned. We find that the subset ``train-clean-100`` of   Librispeech is enough for our study, so we constructed the LS-100 dataset using partial samples from the Librispeech as the source materials. To be specific, we first concatenate all the speakers' speech clips into a long speech, and then select the 100 speakers with the longest duration to cut their voices into two second speech. You can download the Librispeech from [here](https://www.openslr.org/12/).

1. Download [dataset](https://www.openslr.org/resources/12/train-clean-100.tar.gz) and extract the files.

2. Transfer the format of audio files. Move the script ``normalize-resample.sh`` to the root dirctory of extracted folder, and run the command ``bash normalize-resample.sh``.

3. Construct LS-100 dataset.
   
   ```
   python data/LS100/construct_LS100.py --data_dir DATA_DIR --duration_json data/librispeech/spk_total_duration.json --single_spk_dir SINGLE_SPK_DIR --num_select_spk 100 --spk_segment_dir SPK_SEGMENT_DIR --csv_path CSV_PATH --spk_mapping_path SPK_MAPPING_PATH
   ```
   
## Code

```bash
pip install -r requirements.txt
```
Training
```
python train.py -project stdu -dataroot DATAROOT -dataset librispeech -config ./configs/stdu_LS-100_FCIAC -gpu 0
python train.py -project stdu -dataroot DATAROOT -dataset nsynth-100 -config ./configs/stdu_nsynth100_FCIAC.yml -gpu 0
python train.py -project stdu -dataroot DATAROOT -dataset FMC -config ./configs/stdu_fmc89_FCIAC.yml -gpu 0
```

## Contact

Yanxiong Li (eeyxli@scut.edu.cn) and Guoqing Chen (1695390037@qq.com)

School of Electronic and Information Engineering, South China University of Technology, Guangzhou, China
