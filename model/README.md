# Run the model code
```
cd model
```
Don't forget to connect with wandb as desctibed in the setup-readme.
## Baseline
The baseline code was run by using wandb to start many runs with changing combinations of hyperparameters. You need to initialize a sweep with 
```
wandb sweep --project <propject-name> train_model_baseline_sweep.yaml
```
and then run the command that is outputted.

For checking with the german pre-trained model, replace the file in the `train_model_baseline_sweep.yaml` config.

## Train /Fine-tune
This is also done by wandb to restart 20 times with a different seed for a different training-evaluation dataset combination.
 ```
wandb sweep --project <propject-name> train_model_llama_sweep.yaml
wandb sweep --project <propject-name> train_model_mixed_sweep.yaml
```

## Testing + calibration check

Just run the script 
```
python text_model.py
```
Change the dataset-composition and chosen model in the script `test_model.py` here: 
```
CASE_TESTSET = ['hw','mg_llama','mg_mixed','mg_gpt_special','mg_mistral_special'] # ['hw','mg_llama','mg_mixed','mg_gpt_special','mg_mistral_special']# 
CASE_MODEL = 'llama' #['llama','mixed']
```

The calibration is triggered in the script as well.