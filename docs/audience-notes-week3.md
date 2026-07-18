# Audience Notes — Week 3 (M3P1: Experiment Review & Model Selection)
---

## Quinn — CV Colon Screening

### Summary
Quinn's project trained one model, SegResNet, as it balanced accuracy with trust, despite having a model that could get a higher accuracy. Whole-box ROI, clipping the image of the pancreas, helped to lessen the over-predictions. The biggest break through was in the dice metrics, where more data was introduced, raising the specificity score from 0.26 to 0.52. This was aided by raising the case count from 95 to 1412 for training. Right now it detects around 90% of healthy.

### Question I asked
The model say 0.52 accuracy, what is the 90% measuring?

### Assessment
Overall this looks really good, and seeing that an increase in data can raise the specificity score so much, even above the target for next week, is reassuring. The model choice is good too under the explanation of not wanted to clear people as healthy when they're not.

---

## Gracie — Earthquake Risk Forecasting

### Summary
Gracie looked through multiple different features for their model, and through the first pass realized that some features were adding noise. Train/Test split is 72% train, 11% validate, and 17% test. The main metrics looked at were PR-AUC, and Precision. The baseline model used was a Logistic Regression model. Per region modelling provided too little metrics due to the minimal data. Random Forest beat the baseline but struggled with overfitting. Experiments included training off geographic location and one model per location. In the end, the base model was chosen due to lack of overfitting, passing leakage checks, and knowing which features are causing predictions.

### Question I asked
When testing individual models for regions, did you test on combined data? Or region specific data? Was there a lack of data per region causing the lower scoring?

### Assessment
The project seems to be going in a good direction. Gracie said that they still want to test out XGBoost and see what that offers them, while still trying to avoid leakage. I think it's a good choice, but I probably would have looked into the Random Forest a bit more due to the accuracy, however, given data leakage concerns with earthquake predictions, it's a good choice.

---

## Ted — Hand Gesture Game Control

### Summary
The original scope was for the camera to recognize a gesture, and move the curser to a pre-determined spot and click. This changed to live hand tracking movement, where the cursor moves live with the hand position, and click based on a fist. MediaPipe model allowed the camera to detect, crop, and classify the gesture, reading up to 92% accuracy. Then the strategy changed to cursor + binary click in the attempt to bring the accuracy closer to 1. Going into next week, Ted wants to retrain once more to try and remove ambiguity between a fist and a palm, due to an inbetween having trouble guessing in the current build. Into week 4, he wants to test against the game, and potentially see control outside of the game.

### Questions I asked
Are you doing any image preprocessing to improve accuracy? Such as color boosting, attempting to bring out edges more?
Have you considered being able to use two hands at once for interface?

### Assessment
Overall I think it's a great position with a pretty accurate model stack. Being able to accurately detect the recquired gestures and movements needed for the end goal. There's also a solid plan going into next week with refinements to the gesture detection. The model choice is valid with the accuracy and getting the desired results.

---