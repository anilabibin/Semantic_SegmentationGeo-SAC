# Semantic_SegmentationGeo-SAC
Geo-SAC is a geometry-driven framework for LiDAR point cloud semantic segmentation that combines adaptive multi-scale primitive extraction, structure-aware superpoint generation, multi-graph learning, and graph transformers to achieve robust and accurate scene understanding.
# Adaptive Primitive Graph Transformer for 3D Point Cloud Semantic Segmentation

## Overview

This repository presents an **Adaptive Primitive Graph Transformer with Occupancy-Guided Structural Consistency Learning** for semantic segmentation of 3D point clouds.

The proposed framework integrates multi-scale geometric feature extraction, structure-aware superpoint generation, structural boundary preservation, multi-graph construction, graph attention, and occupancy-guided structural consistency learning to achieve geometry-aware semantic segmentation.

Unlike conventional point-wise classification approaches, the framework models relationships between local geometric structures and neighbouring regions through multiple graph representations.

## Proposed Methodology

The framework consists of the following major components:

1. **Adaptive Multi-Scale Primitive Extraction:** Extracts local geometric descriptors at multiple neighbourhood scales, including linearity, planarity, sphericity, and curvature using covariance and eigenvalue analysis.

2. **Structure-Aware Superpoint Generation:** Groups geometrically similar and spatially connected points into meaningful structural regions to reduce computational complexity.

3. **Structural Boundary Preservation:** Identifies structural boundaries using geometric differences, normal variations, and height information.

4. **Boundary-Aware Graph Refinement:** Refines graph connectivity by incorporating structural boundary strength to preserve object and region boundaries.

5. **Multi-Graph Construction:** Constructs three complementary graph representations:
   - Geometric Graph
   - Topological Graph
   - Contextual Graph

6. **Adaptive Primitive Graph Transformer:** Learns relationships between superpoints through graph-based multi-head attention, message aggregation, and boundary-aware feature refinement.

7. **Occupancy-Guided Structural Consistency Learning:** Incorporates auxiliary prediction branches for occupancy, primitive structure, boundary prediction, and cross-view consistency.

8. **Overall Optimization:** Combines semantic classification loss with auxiliary structural losses to optimize the proposed framework.

9. **Semantic Segmentation:** Produces semantic predictions for the input 3D point cloud.

 Framework Architecture

The overall methodology follows this processing sequence:

Raw 3D Point Cloud
        |
        v
Adaptive Multi-Scale Primitive Extraction
        |
        v
Structure-Aware Superpoint Generation
        |
        v
Structural Boundary Preservation
        |
        v
Boundary-Aware Graph Refinement
        |
        v
Multi-Graph Construction
        |
        v
Adaptive Primitive Graph Transformer
        |
        v
Occupancy-Guided Structural
Consistency Learning
        |
        v
Overall Multi-Task Optimization
        |
        v
Semantic Segmentation Output


## Repository Structure


Adaptive_Primitive_Graph_Transformer/
│
├── data/
│   ├── dataset_loader.py
│   ├── preprocessing.py
│   └── augmentation.py
│
├── features/
│   ├── multiscale_features.py
│   └── covariance.py
│
├── segmentation/
│   ├── superpoint_generation.py
│   └── primitive_grouping.py
│
├── graph/
│   ├── boundary_detection.py
│   ├── boundary_refinement.py
│   └── multigraph.py
│
├── models/
│   ├── primitive_graph_transformer.py
│   ├── consistency_heads.py
│   ├── losses.py
│   └── segmentation_model.py
│
├── training/
│   ├── train.py
│   └── evaluate.py
│
├── results/
│   ├── visualization.py
│   └── metrics.py
│
├── checkpoints/
│
├── config.py
├── main.py
└── README.md


## Module Description

| Directory / File | Description |
|---|---|
| `data/` | Dataset loading, preprocessing, and augmentation |
| `features/` | Multi-scale geometric feature extraction and covariance analysis |
| `segmentation/` | Superpoint generation and primitive grouping |
| `graph/` | Boundary detection, graph refinement, and multi-graph construction |
| `models/` | Graph transformer, consistency heads, loss functions, and segmentation network |
| `training/` | Model training and evaluation |
| `results/` | Performance metrics and 3D visualization |
| `config.py` | Configuration parameters |
| `main.py` | Main execution entry point |

## Technologies and Dependencies

The implementation uses Python and the following libraries:

- Python 3.10
- PyTorch
- NumPy
- Open3D
- Scikit-learn
- laspy



For GPU acceleration, install the PyTorch version compatible with your CUDA environment using the official PyTorch installation instructions.

## Dataset Preparation

The framework is designed for 3D point cloud semantic segmentation.

The input point cloud should be preprocessed to generate:

- 3D point coordinates.
- Multi-scale geometric features.
- Superpoint representations.
- Graph connectivity information.
- Semantic ground-truth labels.
- Occupancy and primitive targets.
- Structural boundary annotations.



Each processed scene should contain the following fields:

| Field | Description |
|---|---|
| `features` | Superpoint feature matrix |
| `edge_index` | Graph connectivity |
| `edge_features` | Geometric, topological, contextual, boundary, and positional attributes |
| `semantic` | Semantic ground-truth labels |
| `occupancy` | Occupancy targets |
| `primitive` | Geometric primitive labels |
| `boundary` | Structural boundary targets |
| `points` | Original point coordinates |
| `point_to_superpoint` | Mapping from original points to superpoints |

The auxiliary targets must be generated using the corresponding ground-truth annotations or a documented supervision procedure.

## Model Configuration

The initial model configuration includes:

| Parameter | Default value |
|---|---:|
| Hidden dimension | 128 |
| Attention heads | 4 |
| Transformer layers | 3 |
| Dropout | 0.1 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Training epochs | 100 |
| Number of semantic classes | 5 |

These are initial experimental settings and can be modified according to the dataset and computational resources.

## Model Training

After preparing the training and validation datasets, execute:


python -m training.train \
    --train_dir data/train \
    --val_dir data/validation \
    --epochs 100 \
    --num_classes 5

The training procedure includes:

- Graph-based feature learning.
- Semantic classification.
- Occupancy prediction.
- Primitive prediction.
- Structural boundary prediction.
- Cross-view consistency learning.
- Multi-task loss optimization.



## Model Evaluation

Evaluate the trained model using an independent test dataset:



The evaluation module calculates the following metrics:

- Overall Accuracy (OA)
- Mean Accuracy (mAcc)
- Mean Intersection over Union (mIoU)
- Precision
- Recall
- F1-score

Evaluation results are saved in JSON format.

## 3D Visualization

The predicted semantic labels can be visualized using Open3D.

The visualization module supports:

- 3D semantic point cloud visualization.
- Ground-truth and predicted segmentation comparison.
- Semantic colour mapping.
- Export of classified point clouds in PLY and LAS formats.

## Semantic Classes

The semantic classes depend on the selected dataset and its annotation scheme.

An example five-class configuration is:

| Class ID | Semantic Class |
|---|---|
| 0 | Building |
| 1 | Tree |
| 2 | Road |
| 3 | Car |
| 4 | Terrain |

The class mapping should be adapted to the dataset's original annotation definitions.

## Experimental Evaluation

The framework can be evaluated in terms of:

**Semantic segmentation performance**
- OA
- mAcc
- mIoU
- Precision
- Recall
- F1-score

**Structural performance**
- Boundary Precision
- Boundary Recall
- Boundary F1-score
- Occupancy prediction accuracy

**Computational performance**
- Training time
- Inference time
- GPU memory consumption
- Processing throughput
- Graph construction time

