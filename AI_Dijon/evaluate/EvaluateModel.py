#evaluate the deep learning model
def evaluateModel(model, dijon_train, label_train, dijon_test, label_test) -> str:
    model_eval = ""
    #evaluation in train dataset
    eval_train = model.evaluate(dijon_train, label_train)
    model_eval +="taux de pertes train : "+str(eval_train[0]*100) + " %\n"
    model_eval+="accuracy train : "+str(eval_train[1]*100) + " %\n"
    model_eval+="erreure absolue moyenne train : "+str(eval_train[2]*100) + " %\n"
    #evaluation in test dataset
    eval_test = model.evaluate(dijon_test, label_test)
    model_eval+="taux de pertes test : "+str(eval_test[0]*100) + " %\n"
    model_eval+="accuracy test : "+str(eval_test[1]*100) + " %\n"
    model_eval+="erreure absolue moyenne test : "+str(eval_test[2]*100) + " %"
    return model_eval