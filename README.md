# Polish Border Crisis :poland: :belarus:

![poster-1](https://user-images.githubusercontent.com/43317648/187998123-15b784fb-e95c-47cb-937a-7576770a43fa.jpg)

# Code structure of the project

All the source code is located in the `src/` directory. Inside there are a bunch of subdirectories containing the code for different stages of the project:

- `src/scraper/` - the web scraper written using the [stweet](https://github.com/markowanga/stweet) library
- `src/data-processing/` - downloading the dataset from the database and sampling the data
- `src/annotate_test/` - calculating the annotators agreement
- `src/classify/` - implementation of different classifiers
- `src/data-analysis` - analysis of the results
