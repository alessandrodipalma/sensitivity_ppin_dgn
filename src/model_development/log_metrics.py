from torchmetrics.functional import accuracy, specificity, auroc, recall, precision, f1_score, matthews_corrcoef
from torch import tensor

def log_all(model_instance, set_name: str, out: tensor, ground_truth: tensor, loss, task: str ='binary', num_classes: int =None):
    model_instance.log_dict({f'{set_name}_loss': loss,
                        f'{set_name}_acc': accuracy(out, ground_truth, task=task, num_classes=num_classes),
                        f'{set_name}_spec': specificity(out, ground_truth, task=task, num_classes=num_classes),
                        f'{set_name}_auroc': auroc(out, ground_truth, task=task, num_classes=num_classes, average='weighted'), 
                        f'{set_name}_recall': recall(out, ground_truth, task=task, num_classes=num_classes),
                        f'{set_name}_precision': precision(out, ground_truth, task=task, num_classes=num_classes),
                        f'{set_name}_f1': f1_score(out, ground_truth, task=task, num_classes=num_classes),
                        f'{set_name}_mcc': matthews_corrcoef(out, ground_truth, task=task, num_classes=num_classes)},
                        batch_size=len(ground_truth),
                        sync_dist=True)
