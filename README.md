# Spinach_Disease_Classification
This is a classification model done with the help of ResNET 50 . The accuracy is a bit low but can be improved on adding some more dataset or on training for more epochs. 

ResNet (Residual Network) was introduced in 2015 by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun in their paper titled "Deep Residual Learning for Image Recognition" at the Computer Vision and Pattern Recognition (CVPR) conference. ResNet addressed a critical challenge in training deep neural networks: the vanishing/exploding gradient problem, which made it difficult to train very deep networks effectively.

Before ResNet, deeper networks often performed worse than shallower ones due to degradation problems, where deeper layers failed to learn meaningful representations, leading to higher training errors.

The key innovation in ResNet is the use of skip connections (or residual connections), which allow information to bypass one or more layers. This design helps address the vanishing gradient problem and enables efficient optimization of very deep networks.
y=F(x,{Wi})+x
x is the input.
𝐹(𝑥,{𝑊𝑖}) represents the transformations (e.g., convolutions, batch normalization, activations).
y is the output after adding the input back.

Here the necessary imports are torch library from pytorch. torch.optim is used for optimization , Dataloader is used for loading the data,transforms is used for data preprocessing and used for data transformation, Matplotlib is used for plotting the loss function ,accuraccy score,confusion matrix,etc.

I have connected my google drive to it so as the dataset doesn't get vanished away when the runtime is disconnected.

A convention followed to check if cuda toolkit is present in the system or not. Although I don't have Cuda toolkit to train my model on GPU I have written this (a good practice)

Transformations are applied to the dataset for training data as well as validation data

Splitting the dataset into Training, Validation , Testing where the dataset is dived into Training set size (Augmented + Original(33.33%)): 6008, Validation set size: 201 ,Test set size: 201 .. Here I got a dataset from  ( DOI : 10.17632/n56pn9fncw.2).They already had Augmented dataset as well as the original dataset. I used the whole Augmented dataset and divided the original dataset into 3 parts (training, validation, testing)  

The hyperparameters present are batch_size,num_epochs,learnin_rate

Then ResNET 50 model architecture was defined as I made a class of ResNetClassifier which inherits all the properties of super class(ResNet50Classifier) with no prelearned weights(weights=None). The num_classes are taken 3 as I want to classify into 3 classes.

Then I installed tqdm progress bar to check for the progress in training.

Then I put the defined class ResNet50Classifier(with specifying no of classes as 3 as parameter) into model. Then I transferred the model to my device for training.

Then I called the loss function(cross entropy loss) and the optimization algorithm (ADAM) .. Adam-this uses Stochastic gradient descent algorithm for optimization.

Then I defined the training model and initialized the loss function and optimizer and initialized arrays for storing the training losses,validation loss , etc.

Then I Trained the model for classification under tqdm progress bar , which helps in visualizing the progress in training. Here the model will be trained for 5 epochs. The model will process a batch of 32 samples from the dataset simultaneously.

The validation accuracy came as 82.09% in the end . This can be increased by adding some more datasets or by training again, starting from our prelearned weights .

Then I have plotted the results of precision score, accuracy ,loss function,etc.

Then I saved the model prelearned weights in .pth file which is present in the current repository as 'resnet50_Spinach_Weights.pth'

Then I loaded the model later with the help of resnet50_Spinach_Weights.pth... for testing...which gave an accuracy of 81.59%.

I plotted the confusion matrix which gave 164 correct result for Anthracnose out of 201, and it wrongly classified Anthracnose disease as Algal leaf for 29 samples and 8 samples for Healthy

The error warning of  ,"/usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1531: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior._warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))"         ......  occurs because Scikit-learn's classification metrics (e.g., recall, precision, or F1-score) require some samples for each label in the dataset to calculate these metrics. When one or more labels in the dataset do not have any true samples (i.e., none of the ground truth labels are assigned to that class), the metric becomes undefined..This can be cured if we use Zero division or a better way is if I choose some samples from anthracnose,algal and healthy randomly in same propertion.

The below line is the reference to the dataset, Make sure to cite when you use this dataset....The DOI for the research paper as well as dataset...is ............ (DOI : 10.17632/n56pn9fncw.2)
Dataset from : Rahman, Mushfiqur; Mukherjee, Anirban ; Shanto , Md Hasibul Hasan  (2023), “Malabar Spinach dataset for diseases classification using deep learning approach”, Mendeley Data, V2, doi: 10.17632/n56pn9fncw.2
