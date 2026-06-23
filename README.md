# 🔬 LabGraph: Dynamic Scene Graph Construction, Incremental Updates, and Landmark-Centric Visual Reasoning

LabGraph is a research-oriented framework for dynamic scene understanding that integrates **Scene Graph Generation (SGG)**, **Incremental Scene Graph Updates**, **Pixel-Level Localized Change Detection**, **K-Landmark Spatial Reasoning**, **K-Hop Landmark-Centric Reasoning**, and **Visual Grounding** into a unified pipeline.

The framework is designed for environments where scenes evolve over time and where spatial reasoning must remain robust to object additions, removals, movements, and appearance changes.

---

## 🚀 Key Features

* Dynamic Scene Graph Generation from images
* Incremental Scene Graph Maintenance
* Pixel-Level Localized Change Detection
* Object Addition and Removal Detection
* Object Motion Analysis
* Attribute-Level Change Detection
* K-Landmark Selection for Stable Spatial Anchors
* K-Hop Landmark-Centric Reasoning
* Landmark-Based Visual Grounding
* Natural Language Spatial Query Answering
* Interactive Streamlit Web Application

---

## 🏗️ System Pipeline

1. Object Detection and Recognition
2. Scene Graph Construction
3. Localized Change Detection
4. Incremental Scene Graph Update
5. K-Landmark Selection
6. K-Hop Landmark-Centric Reasoning
7. Visual Grounding
8. Query Answering

---

## 📊 Scene Graph Construction

The framework constructs structured scene graphs where:

* Nodes represent detected objects.
* Edges represent spatial relationships.
* Attributes store semantic and visual properties.

This representation enables efficient reasoning over complex scenes.

---

## 🔄 Incremental Scene Graph Updates

Unlike traditional approaches that rebuild the entire graph after every modification, LabGraph performs incremental updates by:

* Detecting added objects
* Detecting removed objects
* Detecting moved objects
* Detecting attribute modifications

Only affected graph components are updated, significantly reducing computational overhead.

---

## 🎯 K-Landmark Spatial Reasoning

To improve robustness, LabGraph introduces landmark-centric reasoning.

Selected landmarks act as stable spatial anchors that:

* Remain reliable across scene modifications
* Provide broad spatial coverage
* Support interpretable reasoning chains

Example:

> The glasses are right of the student, which is left of the table, which is near the microscope.

---

## 🖼️ Visual Grounding

LabGraph combines scene graph reasoning with visual grounding by:

* Highlighting queried objects
* Highlighting selected landmarks
* Generating human-readable spatial explanations

This allows users to visually verify generated reasoning paths.

---

## 🌐 Interactive Web Application

The project includes a Streamlit-based web interface that supports:

* Uploading original images
* Uploading scene annotations
* Uploading modified images
* Interactive scene graph visualization
* Query answering using multiple reasoning modes
* Output visualization and evaluation

---

## 📁 Project Structure

```text
LabGraph/
│
├── attributes/
├── change_detection/
├── detection/
├── evaluation/
├── graph/
├── models/
├── reasoning/
├── relations/
├── utils/
├── webapp/
│
├── main.py
├── config.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Jitmalla45/LabGraph.git

cd LabGraph

conda create -n labgraph python=3.10

conda activate labgraph

pip install -r requirements.txt
```

---

## ▶️ Running the Web Application

```bash
conda activate labgraph

streamlit run webapp/app.py
```

---

## 📈 Dataset Evaluation

The framework supports automated evaluation across multiple reasoning strata:

* Single-Hop Relational
* Multi-Hop Relational
* Landmark-Based Reasoning
* Visual Grounding Evaluation

Evaluation results are automatically generated and stored for further analysis.

---

## 🔗 Repository

GitHub Repository:

[https://github.com/Jitmalla45/LabGraph](https://github.com/Jitmalla45/LabGraph)

---

## 👨‍💻 Author

**Jit Malla**
Int. MS–PhD Student
School of Mathematical & Computational Sciences (SMCS)
Indian Association for the Cultivation of Science (IACS)

---
