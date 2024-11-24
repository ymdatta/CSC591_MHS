# CSC591_MHS

1. 591_Project.py: Generates the pickle files which contain feature data for all IDs.

In case of training data, if an ID is positive, data is augmented to rebalance the data.
So, the way we would generate features is dependent on if an ID is Covid+/Covid-.

For each ID, generate both training featurs and testing features and store them in the
pickle files 'train_v1.pickle' and 'valid_v1.pickle' respectively.

2. 591_project_v1.py: Takes in the pickle files, performs cross validation and produces
the results.
