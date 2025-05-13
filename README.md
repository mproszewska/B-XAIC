# **B-XAIC Dataset**  
**Benchmarking Explainable AI Using Compound Data for GNNs**

---

## 🛠️ Setup

### 1. Create the Conda Environment
```bash
conda env create -f environment.yaml
conda activate bxaic
````

### 2. Download the Dataset

Download the dataset from [Hugging Face](https://huggingface.co/datasets/mproszewska/B-XAIC) and unpack it into the `data/` directory.

### 3. Install ProtGNN

Clone and set up [ProtGNN](https://github.com/zaixizhang/ProtGNN) inside the `libs/` directory:

```bash
cd libs/
git clone https://github.com/zaixizhang/ProtGNN
```
Follow any installation/setup instructions provided in the ProtGNN repo.

---

## 🧪 Evaluation Pipeline

### 🔹 Train a GNN model
Train a standard GCN/GAT/GIN model
```bash
python train_model.py --model_type GIN --task indole --split 0 --save_path model.pt
```
or train a ProtGNN version of GCN/GAT/GIN model
```bash
python train_protgnn.py --clst 0.1 --sep 0.05 --model_type GIN \
  --task indole --split 0 --readout sum --save_path model.pt
```

### 🔹 Extract Explanations

```bash
python extract_explanations.py --model_path model.pt --save_path explanations.pt \
  --explainer_type Saliency --explanation_type phenomenon \
  --node_mask_type attributes --edge_mask_type none
```

### 🔹 Evaluate Explanations

```bash
python evaluate_explanations.py --explanations_path explanations.pt \
  --mask_to_eval node --save_path results.pt
```