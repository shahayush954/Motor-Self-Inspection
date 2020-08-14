# Motor-Self-Inspection

In the field of Car Insurance, the applicants car need to be investigated for any previous damages which is done by manual checking. The aim of this internship was to eliminated this manual intervention
by auto detecting any damages present in the given car image. I did this project during my internship with DHFL General Insurance. To summarize I :

  1. Created a Model for Motor Self Inspection to automate the process of inspecting cars by Policy Makers.
  2. Conducted testing with 2-3 Algorithms to find the best accuracy of over 65%.
  3. Conducted Data preparation and cleansing for more than 800 images.
  4. Developed a basic portal in Django for using the system.
  
We carried out training for five types of damages which were broken windsheild, dents, cracks, bumper damages, and completed distorted cars. The predictions works in the following
way: We had five weights file(model) after completeing the training process, Each image is taken and is predicted for any damages using these five weights file.(Note: The repository
does not contain the weights files as each file has a size of over 250MB in total making it over 1GB).
We carried out custom training using Matterports Custom Mask RCNN Skeleton. (You can refer this link for your own custom training of Mask RCNN -> Link: https://github.com/matterport/Mask_RCNN)
  
## Directory Structure

  1. The images folder should contain images on which you need to carry out detection.
  2. The mrcnn folder contains all files related to Mask RCNN algorithm.
  3. The maskrcnn_predict.py contains the code for carrying out the prediction on the images.
  
## Download and Usage

  1. Clone this repository to your own system.
  2. Carry out your custom training on your own images by following the steps provided in the above link.
  3. After completing the training you'll have the weights files. Create a folder named "weights" in the clonned folder of this repository. Place all the weights file in this newly created folder.
  4. Place the image you want to caryy out your prediction on, in the images folder.
  4. In your terminal navigate to this folder, and run the command: ```python maskrcnn_predict.py --image images/image name of your desired image```.
