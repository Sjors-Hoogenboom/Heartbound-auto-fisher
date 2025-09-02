
# Function of the program:

This is an automatic fishing bot on discord for the heartbound discord server.

Firstly, you calibrate the region of the screen you want to use, hover your mouse over the left corner, and it takes that position as the first point, then hover your mouse over the bottom right corner and that will be the second point, establishing the chat region 

it types /fish (or any chosen message) and checks for trigger words, if the trigger words are spotted it will resend a /fish message

After a /fish there's a delay (set to 15 seconds by default), after that delay it will make a snapshot of your screen every 5 seconds to check for trigger words


# How to use:

You'll need tesseract for python, for windows it can be downloaded from the github page: https://github.com/UB-Mannheim/tesseract/wiki

You'll need an IDE like pycharm or Visual Studio to run the code