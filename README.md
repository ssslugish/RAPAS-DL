# RAPAS-DL
Deep learning-assisted ensemble data assimilation for atmospheric emission inversion

Overview
RAPAS-DL combines a deep learning (DL) surrogate with ensemble square root filter (EnSRF) emission updating and CMAQ atmospheric state propagation. The surrogate predicts concentration responses to emission perturbations, reducing the cost of repeated chemical transport model ensemble simulations.

This repository covers the DL training and prediction components. The prediction script is called by an external daily emission inversion workflow and handles input preparation, model inference, and NetCDF output. The complete EnSRF and WRF–CMAQ inversion system is outside the scope of this code release.

Separate surrogate models are used for CO and NO₂ concentration prediction, supporting CO and NOₓ emission inversion, respectively. NO₂ is the predicted concentration species for the NOₓ inversion.

Model
The surrogate uses a residual U-Net with attention gates. Its training objective combines Charbonnier loss, masked structural similarity loss, and a spatial gradient loss weighted by the same adaptive mask.

Runtime environment
The author reports the following software versions:
Component	Version or status
Python	3.9.7
TensorFlow	2.4.1

Input data
The CO and NOₓ inversion applications use different predictor sets. Prepare the complete WRF NetCDF files required by the prediction script. The surrogate also uses emission and background concentration information; the script-specific sources and file layout must be documented alongside the released code.

Before running either model, verify:
- Required input files and variable names.
- Predictor channel order for the selected species.
- Array dimensions, grid alignment, vertical levels, and time selection.
- Variable units and preprocessing or normalization settings.
- Any output rescaling needed to recover physical concentration units.
The exact input specification is pending confirmation from the released scripts. WRF files alone should not be assumed to provide every required model input.

Training
1. Prepare the training and validation inputs and corresponding CMAQ concentration targets for the selected species.
2. Set the input paths, output directory, and species-specific settings used by the training script.
3. Run the training script in the configured environment.
Command template — replace the filename with the actual script name:
python Final_train_script.py
Save the trained model and any preprocessing parameters needed by the prediction script. Exact script names and configuration locations will be added after the code is organized.

Prediction
1. Prepare the input NetCDF files and any additional emission and background concentration inputs required by the script.
2. Select the trained model and preprocessing settings for the appropriate species.
3. Set the prediction date, input locations, and output location in the manner supported by the script.
4. Run the prediction script directly or call it from the external daily inversion workflow.
Command template — replace the filename with the actual script name:
python Predict_Script.py
These templates do not imply that command-line arguments are implemented. Use the configuration mechanism in the actual scripts.

Output
Predicted concentration fields are combined into a NetCDF file for use by the external inversion workflow. The output contains concentration predictions, not posterior emission estimates.
Output filenames, variable names, units, dimension order, and the organization of species and ensemble members will be documented from the prediction script.

Trained models and example data
Download locations for trained weights, preprocessing parameters, and example inputs have not yet been specified. Inference requires a compatible trained model and its associated preprocessing settings.

Related manuscript
A Hybrid Deep Learning and Data Assimilation Framework Enables Rapid Top-Down Inversion of Multipollutant Emissions
Status: submission in progress.

The final publication citation and the archived software version DOI will be added when available.
Contact
Honglin Zhuang
Jiangsu Provincial Key Laboratory of Geographic Information Science and Technology, International Institute for Earth System Science, Nanjing University, Nanjing, 210023, China
Email: zhuanghonglin1213@163.com
