# Predict
for k in range(100):
    Yp = [0] * n_tags
    Yk = Y[k,:].to_array()

    for tag in range(n_tags):
        Yp[tag]=clf[tag].predict(X[k,:])


    #Evaluate
    true_positives = sum([a*b for a,b in zip(Yp,Yk)])
    predicted_positives = sum(Yp)
    actual_positives = sum(Yk)
    precision = true_positives / predicted_positives
    recall = true_positives / actual_positives
    F1 = 2 * precision * recall / (precision + recall)

    print('F1[{}]: {}'.format(k,F1))