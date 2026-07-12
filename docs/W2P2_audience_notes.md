Audience Notes Quality (Audience)
Notes for all presentations include a summary, questions asked, and an assessment of whether EDA findings justify the modeling plan — showing analytical thinking.

# Gracie - Earthquake Data
This project still focuses on earthquakes in three regions. California, Greece, and Japan. A lot of the EDA focused on removing leakage and determining magnitude, distance to fault lines, and how often they happen. The chosen features include region, count, days since last, max magnitude, average magnitude, and an activity trend comparing the current week to the prior week. 

With the chosen metrics, the plan is to pull historical data from the API, scrapes recent earthquake data every two to three days, with retraining every month in theory. I believe the current plan is justified given the EDA, with

Are you missing out on any accuracy by not retraining everyday? Does choosing not to retrain everyday have more to do with compute power/time rather than model accuracy?


# Quinn - CV Screening
Quinn's project uses a dataset of 9900 files to train a model on detecing PC from images. It pairs cases and reads headers, it's loaded and preprocesed using image resizers, reorienters, etc, batching, then predicts voxels. The data being used is annotated images of the pancreas, with some having tumors and others being normal. The big thing is the model can identify smaller cases via 3D arrays that humans could struggle to see. More metrics are being looked at to better understand the results, however the early exploration yields already relatively strong results.

Given the EDA and plan, it definitely feels on pace to work out given the implementation plan. Even though the accuracy of the model so far isn't as strong as the comparison, this model is working on a much smaller dataset. Nothing thus far leads me to believe there would be significant issues in this plan.

Could there be inconsistencies or missing columns from datasets taken from other sites or countries, to where your model could determine cases wrong? Do you know if the quality from all the sites and sources is consistent throughout?


# Ted - CV Hand Gestures
The project focuses on taking in multiple pictures of hand gestures, training a model on preprocessed photos, and being able to control a game with a web cam. There are currently two routes being looked at. Either training the model on upclose, cropped images, allowing the user to play up close at a desk, or far away images for a demo where a group is standing further away from the camera. One model for box, one model for gesture.

The implementation plan hasn't changed much at this point aside from models, but there is a lot more to think about. Mainly which model would work best, with Ted believing that either model will work just fine. This doesn't change the overall plan, and I agree that it is still feasible given a model can be trained and work effectively.

Is there a training method, like only up close images or only far images, that you know tends to work better for this use case? Could you potentially train on both up close and far away photos, attaches a bbox, and have it not care if you're up close or far away? 