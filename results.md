# Week-1

## Adding Cross-Attn
HyperParams: \
--dim_he_list 400 200 200 \
--view_list 1 2 3 \
--num_epoch 20000 \
--lr_e 0.0005 \
--lr_c 0.001
### BRCA

#### Baseline:
              precision    recall  f1-score   support

           0       0.49      0.86      0.62        35
           1       0.89      0.87      0.88        39
           2       1.00      0.64      0.78        14
           3       0.84      0.82      0.83       131
           4       0.74      0.45      0.56        44

    accuracy                            0.76       263
    macro avg       0.79      0.73      0.74       263
    weighted avg    0.79      0.76      0.76       263


#### Hierarchial Attn:
              precision    recall  f1-score   support

           0       0.48      0.86      0.62        35
           1       0.94      0.87      0.91        39
           2       0.82      0.64      0.72        14
           3       0.89      0.79      0.83       131
           4       0.74      0.64      0.68        44

    accuracy                            0.78       263
    macro avg       0.77      0.76      0.75       263
    weighted avg    0.81      0.78      0.78       263


### ROSMAP
#### Baseline:
                  precision    recall  f1-score   support

           0       0.78      0.57      0.66        51
           1       0.68      0.85      0.76        55

    accuracy                            0.72       106
    macro avg       0.73      0.71      0.71       106
    weighted avg    0.73      0.72      0.71       106 

#### Hierarchial Attn:
                  precision    recall  f1-score   support

           0       0.79      0.82      0.81        51
           1       0.83      0.80      0.81        55

    accuracy                            0.81       106
    macro avg       0.81      0.81      0.81       106
    weighted avg    0.81      0.81      0.81       106

