from ..model_testers import hybrid_determination_tester
from ..training.trainers import hybrid_determination_trainer

# Train and test all models


# train:        Whether to train the model              (Default: True)
# test:         Whether to test the model               (Default: True)
# passes:       The number of passes to use in testing  (Default: 10)
def hybrid_determination_model(train=True, test=True, passes=10):
    if train:
        hybrid_determination_trainer.train()
    if test:
        hybrid_determination_tester.test(passes=passes)


if __name__ == '__main__':
    hybrid_determination_model(train=True, test=True, passes=10)
