# ITU-ML5G-PS-032-KDDI-UT-NakaoLab-AI
# Introduction of source code
- `conf.yaml`: path environment configuration file
- `1_feature_extract.py`: Extract features from json files into csv files and store them on a daily basis.
- `2_feature_combine.py`: Merge features in the CSV files into training and test sets.
- `3_feature_refine.py`: Refine the features by differentiating the normal and abnormal dataset.
- `4_train.py`: Train and test refined_dataset and print test results.
- `label_extract.py`: Extract labels of events
- `diff_analysis.ipynb`: Analyze the differentiated_dataset(for jupyter notebook).
- `feature_importance_analysis.ipynb`: Feature Importance Analysis(for jupyter notebook).

# Path Environment configuration in conf.yaml
- PATH used in `1_feature_extract.py`
    - `DATA_SET`: The root directory of the dataset, you need to put the label json file under this.
    - `LEARNING_DIR`: The data set for learning of Theme 1. Default path is "DATA_SET/data-for-learning".
    - `EVALUATION_DIR`: The data set for evaluation of Theme 1. Default path is "DATA_SET/data-for-evaluation".
    - `PHYSICAL_SUB_DIR`: Physical subdirectory.
    - `NETWORK_SUB_DIR`: Network subdirectory.
    - `VIRTUAL_SUB_DIR`: Virtual subdirectory.

- PATH used in  `2_feature_combine.py`, `3_feature_refine.py`, `4_train.py`
    - `TRAIN_PATH`: The learning CSV file generated from 1_feature_extract. Default path is "DATA_SET/csv-for-learning/".
    - `TEST_PATH`: The evaluation CSV file generated from 1_feature_extract. Default path is "DATA_SET/csv-for-evaluation/".
    - `CSV_DATA_SET`: The CSV file generated by "2_feature_combine". Default file path is "./csv/dataset.csv".
    - `CSV_TEST_SET`: The CSV file generated by "2_feature_combine". Default file path is "./csv/testset.csv".
    - `CSV_DIFF_DATA_SET`: The CSV file generated by "3_feature_refine". Default file path is "./csv/diff_dataset.csv".
    - `CSV_DIFF_TEST_SET`: The CSV file generated by "3_feature_refine". Default file path is "./csv/diff_testset.csv".

# Program Running Environment
- [Python 3.7.6](https://www.python.org/downloads/release/python-376/)
- [jupyter-notebook 6.0.3](https://jupyter.org/install)

# Required python packages
Please install CMake before install xgboost
- CMake(version:3+)
    - `brew install cmake` for OS X
    - `sudo apt install cmake` for Ubuntu
    - `yum install cmake` for CentOS
   
-  pip3 install pandas==0.24.2
-  pip3 install pyyaml==5.3
-  pip3 install matplotlib==3.3.0
-  pip3 install scikit-learn==0.23.2
-  pip3 install xgboost==1.1.1

# Getting Started
1. Check the Python version：

        $ python3 -V
        
2. Switch to project directory:

        $ cd <Your path>/itu-ml-challenge
        
3. If you have not completed the extraction steps:
Modify the path information in the configuration file.
Make sure `<TRAIN_PATH>`, `<TEST_PATH>` (conf.yaml) these two folders exist. If not, create it with `mkdir`.
Then run: (`Attention: This step will take about 10+ hours to extract all the JSON files into ./dataset/csv-for-learning and ./dataset/csv-for-evluation folders. We have already checked in the above two folders so that we can skip this step. 
        
        $ python3 1_feature_extract.py
            
4. Check that CSV files have been generated under <TRAIN_PATH> and <TEST_PATH>.
Then run:

        $ python3 2_feature_combine.py

5. Check whether `dataset.csv` and `testset.csv` have been generated under `./csv/` in the current directory.
Then run:
        
        $ python3 3_feature_refine.py
        
6. Check whether `diff_dataset.csv` and `diff_testset.csv` have been generated under `./csv/` in the current directory.        
If these two files have been generated, congratulations on the completion of the extraction.
Then run:

        $ python3 4_train.py
        
7. You will see the training and test results printed on the console.
At the same time, you can also use jupyter notebook to analyze the data.

        jupyter notebook
        
Choose to open `diff_analysis.ipynb` and `feature_importance_analysis.ipynb`.

# Labels of Events
- 0: ixnetwork-traffic-start
- 1: node-down
- 2: node-up
- 3: interface-down
- 4: interface-up
- 5: tap-loss-start
- 6: tap-loss-stop
- 7: tap-delay-start
- 8: tap-delay-stop
- 9: ixnetwork-bgp-injection-start
- 10: ixnetwork-bgp-injection-stop
- 11: ixnetwork-bgp-hijacking-start
- 12: ixnetwork-bgp-hijacking-stop
- 13: ixnetwork-traffic-stop
# Performance Evaluation
- Please refer to report ITU-JP-Theme1_UT_workshop.pdf (https://github.com/ITU-AI-ML-in-5G-Challenge/ITU-ML5G-PS-032-KDDI-UT-NakaoLab-AI/blob/master/ITU-JP-Theme1_UT_workshop.pdf)
# Demo
 - Plear find our demonstration video at https://exchange.nakao-lab.org/index.php/s/URHs2lLJeR1zOQQ

