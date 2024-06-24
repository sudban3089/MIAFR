import os
config2 = 'config.yaml'
datasets = ["/path/to/CelebA_HQ"]
def create(config, i):
    for dataset_name, path in zip(["CelebAMaskHQ"], datasets):
        for token in ['wzx']:
            train_subjects = sorted(os.listdir(path))[:100] # we use 100 subjects
            for subject in train_subjects:
                f = open(f"CelebAMaskHQ_attributereg_job_list/{i}.sh", "w+")
                a = f"./attribute_sd.sh {token} {subject} {path} {dataset_name}\n"
                f.write(a)
                f.close()
                i += 1
    return i
i = create(config2, 1)
print(i)
