# Polish Border Crisis :poland: :belarus:

![poster-1](https://user-images.githubusercontent.com/43317648/187998123-15b784fb-e95c-47cb-937a-7576770a43fa.jpg)

# Code structure of the project

All the source code is located in the `src/` directory. Inside there are a bunch of subdirectories containing the code for different stages of the project:

- `src/scraper/` - twitter scraper developed using the [stweet](https://github.com/markowanga/stweet) library
- `src/data_processing/` - processing of the raw tweets data from database
- `src/annotate_test/` - annotators agreement calculations
- `src/classify/` - data classyfication based on sample annotations using different classifiers
- `src/data_analysis` - results analysis
