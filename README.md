# Graph-FINDER: Enabling Data-Driven Dopant Discovery in Phase Change Materials via Automated Text and Figure Extraction


<img width="600" height="400" alt="fig_intro_overview" src="https://github.com/user-attachments/assets/a67a79e6-1033-4d24-9a30-5119747fe35e" />

Graph-FINDER is a multimodal AI framework links literature mining with machine learning–based prediction to establish a scalable foundation for identifying optimal material candidates. In the database creation stage (blue), Graph-FINDER employs natural language processing, computer vision, and large language models to extract textual and graphical information from publications, converting fragmented literature into curated, machine-actionable datasets. These datasets enable application-aware prediction (orange), where models estimate candidate properties, benchmark them against device requirements, and uncover interpretable structure–property relationships. The synthesis and validation stage (green) is included as a prospective extension rather than a contribution of this work, representing future experimental realization and measurement of top-ranked candidates to refine the database. This envisioned feedback loop highlights the broader potential of Graph-FINDER for accelerating functional materials discovery.

# 1. Automated pipeline for isolating curves from published figures, including region detection, axis localization, and text annotation to enable accurate data extraction (crop_roi_from_graph.py)
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/5595a498-b22b-400e-b244-8626cb03d359" />

The process begins with acquiring original images from literature sources (a). Visual feature detection is then used to identify regions likely containing closed rectangles (b). The main plotting areas, typically defined by rectangular regions exceeding a specific area threshold and including the primary axes, are localized and extracted to focus on the core measurement content (c). Subplots and auxiliary axes are removed to reduce visual clutter, ensuring only the relevant data region is retained (d). Finally, textual elements such as legends, labels, and annotations are detected to preserve essential contextual information needed for accurate interpretation of the extracted data (e).

# 2. Extraction framework through reconstruction of numerical data from a sample graph (save_data_point_into_file.py)
<img width="600" height="800" alt="figure5_new" src="https://github.com/user-attachments/assets/003b7399-4aea-4e47-b8c6-48c8490a3c2e" />

 The process begins with a sample graph, where individual curves are separated through binary color masking. A virtual grid is then overlaid to detect and isolate data points corresponding to each curve. The extracted values are digitized into structured numerical tables, which are subsequently plotted to verify accuracy. Finally, the reconstructed plots are combined to reproduce the original figure with high precision.



# 3. Evaluation of prompting and fine-tuning strategies for automated extraction of axes metadata from graph images (save_data_point_into_file.py)
<img width="700" height="700" alt="figure4_new" src="https://github.com/user-attachments/assets/89329722-2143-4fbd-b132-1d30dce1da3f" />

(a) Baseline Prompt: The original instruction is supplied to the out-of-the-box GPT-4 Vision Preview model, yielding separate axis labels with only their minimum and maximum values. (b) Revised Prompt: An enhanced prompt emphasising full coverage of the $y$-axis range (from graph box bottom to top) is applied to the same GPT-4 model, resulting in improved identification of axes ranges and legend colours with estimated bounds. (c) Fine-Tuned GPT-4: A GPT-4 model is fine-tuned on a bespoke training set of 50 graph images (10 epochs, batch size 1 and training loss 0.0072) to directly output the axes labels, their exact min/max values, and legend information in a single specified format. Output examples for each method are shown at the bottom, illustrating progressive gains in completeness and precision of metadata extraction.
