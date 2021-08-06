#evaluate the deep learning model
from scipy.stats.stats import mode


def evaluateModel(model, dijon_train, label_train, dijon_test, label_test) -> str:
    #evaluation in train dataset
    eval_train = model.evaluate(dijon_train, label_train)
    model_eval ="Pertes sur le train : "+str(eval_train[0]) + "\n"
    model_eval+="erreure absolue moyenne (MAE) sur train : "+str(eval_train[1]) + "\n"
    #evaluation in test dataset
    eval_test = model.evaluate(dijon_test, label_test)
    model_eval+="Pertes sur le test : "+str(eval_test[0]) + "\n"
    model_eval+="erreure absolue moyenne (MAE) sur test : "+str(eval_test[1]) + "\n"
    model_eval+="\n"
    model_eval+="Votre model généralise." if abs(eval_train[1]*100 - eval_test[1]*100) <= 5 else "Votre model a tendance à suraprendre."
    model_eval+='\n'
    return model_eval