The dataset includes authentic satellite images of Tacoma (2016 pieces), Seattle (1008 pieces), and Beijing (1008 pieces), as well as fake satellite images of Tacoma in the visual pattern of Seattle (2016 pieces) and in the visual pattern of Beijing (2016 pieces).


The codes include two parts -- "extract_features.py" for feature extracting and "train_antiFake_model.py" for antiFake model training and evaluating.

We didn't divide sattelite images to traning and testing datasets in the data folder, instead, we did this in our python code.

The recurring results may be slightly different from the report in the paper, but they are generally consistent. This difference may come from the randomness of division of training set, validation set, and test set.



