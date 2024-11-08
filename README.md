# Deception-RepE

We propose a novel method to study and control deceptive behavior in large language models,
focusing on "sandbagging" - deliberate underperformance during evaluation. Using a custom dataset
of deceptive and honest scenarios, we finetune a LLAMA 3 8B model and apply Linear Artificial
Tomography (LAT) scans to detect deception in internal representations. Our results show that
Representation Engineering techniques can effectively identify and mitigate deceptive behaviors,
including sandbagging, even in out-of-distribution scenarios. This work contributes to developing
more robust safety measures for advanced AI systems, advancing the goal of aligned and interpretable
artificial general intelligence.

The dataset for Zero shot is linked here: https://huggingface.co/datasets/Avyay10/DeceptionQAnew

The dataset used for Sandbagging is linked here: https://huggingface.co/datasets/Avyay10/DeceptionLabelsFinal

The finetuned model is linked here: https://huggingface.co/Avyay10/llama-3-finetuned-final
