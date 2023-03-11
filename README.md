### Abstract
This project was created for http://school-slon.ru/ hackathon. As part of the assignment model for determining time period painting is belonged was designed. 


### Proposed approach
Firtly we downloaded a large amount of data with paintings from 
https://www.wga.hu
For better class balance we decided to unite classes beloged to 201-1300 гг  
![](https://github.com/valer1435/painting_dating/blob/master/README/data.png)  
For solving task this architecture was chosen:
![](https://github.com/valer1435/painting_dating/blob/master/README/model_architecture.png)  
- Vgg19 network up to conv_5_1 (1)
- Applying gram matrix (2) transformation for (1) 
- Filtering 8000 features from (2)
- SVC with 13 classes


# Results

On test set we've got sush results:
- MSE: 4  (it means the model wrong on one century in average)
- F1-score: 0.5  
Confusion matrix:
 ![](https://github.com/valer1435/painting_dating/blob/master/README/results.png)  
