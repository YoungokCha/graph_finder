# Graph-FINDER: Enabling Data-Driven Dopant Discovery in Phase Change Materials via Automated Text and Figure Extraction

## Explore the Graph-FINDER's results in **VO2 dopant database** ➜ [https://github.com/<owner>/VO2-dopant-database](https://wm355.github.io/GraphFinderDatabase/)

<p align="center">
 <img width="600" height="400" display="center" alt="fig_intro_overview" src="https://github.com/user-attachments/assets/a67a79e6-1033-4d24-9a30-5119747fe35e" />
</p>

Graph-FINDER is a multimodal AI framework links literature mining with machine learning–based prediction to establish a scalable foundation for identifying optimal material candidates. In the database creation stage (blue), Graph-FINDER employs natural language processing, computer vision, and large language models to extract textual and graphical information from publications, converting fragmented literature into curated, machine-actionable datasets. These datasets enable application-aware prediction (orange), where models estimate candidate properties, benchmark them against device requirements, and uncover interpretable structure–property relationships. The synthesis and validation stage (green) is included as a prospective extension rather than a contribution of this work, representing future experimental realization and measurement of top-ranked candidates to refine the database. This envisioned feedback loop highlights the broader potential of Graph-FINDER for accelerating functional materials discovery.


1. crop_roi_from_graph.py: Isolating curves from published figures, including region detection, axis localization, and text annotation
<p align="center">
 <img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/12572a36-00fa-498d-b897-9a9a3ad85e45" />
</p>


The process begins with acquiring original images from literature sources (a). Visual feature detection is then used to identify regions likely containing closed rectangles (b). The main plotting areas, typically defined by rectangular regions exceeding a specific area threshold and including the primary axes, are localized and extracted to focus on the core measurement content (c). Subplots and auxiliary axes are removed to reduce visual clutter, ensuring only the relevant data region is retained (d). Finally, textual elements such as legends, labels, and annotations are detected to preserve essential contextual information needed for accurate interpretation of the extracted data (e).



2. save_data_point_into_file.py: Reconstruction of numerical data from the graphs
<p align="center">
 <img width="600" height="800" alt="figure5_new" src="https://github.com/user-attachments/assets/003b7399-4aea-4e47-b8c6-48c8490a3c2e" />
</p>


 The process begins with a sample graph, where individual curves are separated through binary color masking. A virtual grid is then overlaid to detect and isolate data points corresponding to each curve. The extracted values are digitized into structured numerical tables, which are subsequently plotted to verify accuracy. Finally, the reconstructed plots are combined to reproduce the original figure with high precision.


3. new_dopants_generation.py: Dopant–Property Prediction
<p align="center">
 <img width="800" height="400" alt="fig_method_prediction" src="https://github.com/user-attachments/assets/7eefa6ff-4e2b-48b6-9012-ccd74444c40b" />
</p>


A description generator utilized the Graph-FINDER database in combination with the Mendeleev AP to extract detailed chemical and physical descriptors of VO2 samples doped with known elements (e.g., molybdenum). These descriptors include information such as atomic number, oxidation state, ionic radius, and electronegativity. The generated textual descriptions are then transformed into numerical text embeddings using OpenAI’s language model and subsequently fed into a multi-layer perceptron (MLP) regressor. The MLP is trained to predict key metal–insulator transition (MIT) parameters, including the resistance ratio, hysteresis width and transition temperature. Once trained, the MLP regressor is used to evaluate potential candidate dopants. Descriptions of these candidates are processed in the same way as the training data, enabling the model to predict their corresponding MIT properties.
