import ktrain
import timeit
from ktrain import text

# create a function for training roberta and xlnet models
def create_and_train_model(train, val, transformer_model, model_name):
    model = transformer_model.get_classifier()
    model_learner_ins = None
    if model_name == "roberta":
        print("\nCompiling & Training RoBERTa for maxlen=512 & batch_size=6")
        model_learner_ins = ktrain.get_learner(model=model,
                                               train_data=train,
                                               val_data=val,
                                               batch_size=6)
                                               
        print("Model Summary: \n", model_learner_ins.model.summary())
        start_time = timeit.default_timer()
        print("\nFine Tuning RoBERTa on Human Emotion Dataset with learning rate=3e-5 and epochs=3")
        model_learner_ins.fit_onecycle(lr=3e-5, epochs=3)
        stop_time = timeit.default_timer()
        print("Total time in minutes for Fine-Tuning RoBERTa on Emotion Dataset: \n", (stop_time - start_time) / 60)

    elif model_name == "xlnet":
        print("\nCompiling & Training XLNet for maxlen=128 & batch_size=32")
        model_learner_ins = ktrain.get_learner(model=model,
                                               train_data=train,
                                               val_data=val,
                                               batch_size=32)

        print("Model Summary: \n", model_learner_ins.model.summary())
        start_time = timeit.default_timer()
        print("\nFine Tuning XLNet on Human Emotion Dataset with learning rate=2e-5 and epochs=3")
        model_learner_ins.fit_onecycle(lr=2e-5, epochs=3)
        stop_time = timeit.default_timer()
        print("Total time in minutes for Fine-Tuning XLNet on Emotion Dataset: \n", (stop_time - start_time) / 60)

    return model_learner_ins


def check_model_performance(model_learner_ins, class_label_names, model_name):
    print("{} Performance Metrics on Human Emotion Dataset :\n".format(model_name), model_learner_ins.validate())
    print("{} Performance Metrics on Human Emotion Dataset with Class Names :\n".format(model_name),
          model_learner_ins.validate(class_names=class_label_names))
    return None


def save_fine_tuned_model(model_learner_ins, preprocessing_var, model_name):
    if model_name == "roberta":
        predictor = ktrain.get_predictor(model_learner_ins.model, preproc=preprocessing_var)
        predictor.save(
            '../output/roberta-content/roberta-emotion-predictor')
    elif model_name == "xlnet":
        predictor = ktrain.get_predictor(model_learner_ins.model, preproc=preprocessing_var)
        predictor.save(
            '../output/xlnet-content/xlnet-emotion-predictor')
    return None


def load_model(model_name):
    predictor = None
    if model_name == "roberta":
        predictor = ktrain.load_predictor(
            '../output/roberta-emotion-predictor')
        print("RoBERTa model loaded successfully: \n", predictor.get_classes())
    elif model_name == "xlnet":
        predictor = ktrain.load_predictor(
            '../output/xlnet-emotion-predictor')
        print("XLNet model loaded successfully: \n", predictor.get_classes())
    return predictor
