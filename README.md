# Polish Border Crisis :poland: :belarus:

![poster-1](https://user-images.githubusercontent.com/43317648/187998123-15b784fb-e95c-47cb-937a-7576770a43fa.jpg)

# Code structure of the project

All the source code is located in the `src/` directory. Inside there are a bunch of subdirectories containing the code for different stages of the project:

### Scraper
Twitter scraper developed using the [stweet](https://github.com/markowanga/stweet) library is located in `src/scraper/` directory.
  
Running scraper in docker container
```shell
docker build scraper/ -t scraper
docker compose -p twitter_scraper up
```

### Data processing
Scripts for downloading data from MongoDB and raw tweets preprocessing are located in `src/data_processing/` directory.

In order to download data and preprocess tweets run commands:
```shell
python download_data.py --env $path_to_env_file$ --out_dir $out_dir$
python preprocess_data.py --text_col $name_of_text_column$ --in_file  $in_file$ --out_dir $out_dir$
```

To get representative sample of tweets for annotation go through the `sample_data.ipynb` notebook.

### Annotator agreement tests
Annotators agreement tests are located in `src/annotate_test/` directory.

In order to run Cohen Kappa and Krippendorff Alpha agreement tests run command:
```shell
python test_annotations.py
```

### Stance classification
Tweets embedding and classification scripts are located in `src/classify/` directory.

For tweets embedding LaBSE model was used. In order to train LaBASE model and embedd tweets run commands:
```shell
python train_embedding_model.py --embedding_dim $embedding_dim$ --out_dir $out_dir$
python embedd_data --labse_model_path $path_to_trained_model$ --max_seq_length $max_seq_length$ --in_dir $in_dir$ --out_dir $out_dir$
```

Two classifiers are available: MLP and XGboost.  
In order to train MLP classifier on annotated sample and get evaluation results run commands:
```shell
cd train_model/
python mlp.py --model_name $model_name$ --epochs $epochs_num$ --batch_size $batch_size$ --accelerator $accelerator_device$ --params_path $paths_to_model_params_file$ --test_size $percentage_test_size$ --data_path $data_path$ --out_dir $out_dir$
```

For XGBoost classifier run commands:
```shell
cd train_model/
python xgb.py --model_name $model_name$ --params_path $paths_to_model_params_file$ --test_size $percentage_test_size$ --data_path $data_path$ --out_dir $out_dir$
```

To classify all tweets using trained model run commands:
```shell
python classify.py --model_type $model_type(mlp or xgb)$ --model_path $path_to_trained_model$ --params_path $paths_to_model_params_file$ --data_path $data_path$ --out_dir $out_dir$
```

### Data analysis
Jupyter notebooks with data analysis are located in `src/data_analysis/` directory.

Data analysis consists of notebooks:
- `simple_stats.ipynb` - basic statistics and PandasProfiling report.
- `relations.ipynb` - analysis of relationships of the most popoular Polish tweeter users.
- `cartography.ipynb` - analysis of easy, ambiguous and hard to classify tweet cases using [Dataset Cartography](https://github.com/allenai/cartography) library.
- `stance_analysis.ipynb` - analysis of the stance and sentiment over time in different countries.

