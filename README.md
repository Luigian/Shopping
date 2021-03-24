# Shopping

## An AI to predict whether online shopping customers will complete a purchase.

<img src="resources/shopping_output.png" width="1000">

When users are shopping online, not all will end up purchasing something. Most visitors to an online shopping website, in fact, likely don’t end up going through with a purchase during that web browsing session. It might be useful, though, for a shopping website to be able to predict whether a user intends to make a purchase or not: perhaps displaying different content to the user, like showing the user a discount offer if the website believes the user isn’t planning to complete the purchase. How could a website determine a user’s purchasing intent? That’s where machine learning will come in.

The goal in this problem wass to build a nearest-neighbor classifier to solve this problem. Given information about a user — how many pages they’ve visited, whether they’re shopping on a weekend, what web browser they’re using, etc. — our classifier will predict whether or not the user will make a purchase. Our classifier won’t be perfectly accurate — perfectly modeling human behavior is a task well beyond the scope of this proyect — but it should be better than guessing randomly. To train our classifier, we’ll provide use some data from a shopping website from about 12,000 users sessions.

How do we measure the accuracy of a system like this? If we have a testing data set, we could run our classifier on the data, and compute what proportion of the time we correctly classify the user’s intent. This would give us a single accuracy percentage. But that number might be a little misleading. Imagine, for example, if about 15% of all users end up going through with a purchase. A classifier that always predicted that the user would not go through with a purchase, then, we would measure as being 85% accurate: the only users it classifies incorrectly are the 15% of users who do go through with a purchase. And while 85% accuracy sounds pretty good, that doesn’t seem like a very useful classifier.

Instead, we’ll measure two values: sensitivity (also known as the “true positive rate”) and specificity (also known as the “true negative rate”). Sensitivity refers to the proportion of positive examples that were correctly identified: in other words, the proportion of users who did go through with a purchase who were correctly identified. Specificity refers to the proportion of negative examples that were correctly identified: in this case, the proportion of users who did not go through with a purchase who were correctly identified. So our “always guess no” classifier from before would have perfect specificity (1.0) but no sensitivity (0.0). The goal was to build a classifier that performs reasonably on both metrics.

**Supervised Learning**

Supervised learning is a task where a computer learns a function that maps inputs to outputs based on a dataset of input-output pairs.

There are multiple tasks under supervised learning, and one of those is Classification. This is a task where the function maps an input to a discrete output. For example, given some information on humidity and air pressure for a particular day (input), the computer decides whether it will rain that day or not (output). The computer does this after training on a dataset with multiple days where humidity and air pressure are already mapped to whether it rained or not.

This task can be formalized as follows. We observe nature, where a function f(humidity, pressure) maps the input to a discrete value, either Rain or No Rain. This function is hidden from us, and it is probably affected by many other variables that we don’t have access to. Our goal is to create function h(humidity, pressure) that can approximate the behavior of function f. Such a task can be visualized by plotting days on the dimensions of humidity and rain (the input), coloring each data point in blue if it rained that day and in red if it didn’t rain that day (the output). The white data point has only the input, and the computer needs to figure out the output.

<img src="resources/graph_01.png" width="600">

**Nearest-Neighbor Classification**

One way of solving a task like the one described above is by assigning the variable in question the value of the closest observation. So, for example, the white dot on the graph above would be colored blue, because the nearest observed dot is blue as well. This might work well some times, but consider the graph below.

<img src="resources/graph_02.png" width="600">

Following the same strategy, the white dot should be colored red, because the nearest observation to it is red as well. However, looking at the bigger picture, it looks like most of the other observations around it are blue, which might give us the intuition that blue is a better prediction in this case, even though the closest observation is red.

One way to get around the limitations of nearest-neighbor classification is by using k-nearest-neighbors classification, where the dot is colored based on the most frequent color of the k nearest neighbors. It is up to the programmer to decide what k is. Using a 3-nearest neighbors classification, for example, the white dot above will be colored blue, which intuitively seems like a better decision.

A drawback of the k-nearest-neighbors classification is that, using a naive approach, the algorithm will have to measure the distance of every single point to the point in question, which is computationally expensive. This can be speed up by using data structures that enable finding neighbors more quickly or by pruning irrelevant observation.

**Scikit-learn**

There are multiple libraries that allow us to conveniently use machine learning algorithms. One of such libraries is scikit-learn. After importing the libraries, we can choose which model to use. The KNeighborsClassifier uses the k-neighbors strategy, and requires as input the number of neighbors it should consider. Since the algorithm is used often in a similar way, scikit-learn contains additional functions that make the code even more succinct and easy to use.

Now we can train our model on the data set and see if we can predict whether online shopping customers will complete a purchase or not.

------------------------------------

## Implementation

First, open up shopping.csv, the data set provided to you for this project. You can open it in a text editor, but you may find it easier to understand visually in a spreadsheet application like Microsoft Excel, Apple Numbers, or Google Sheets.

There are about 12,000 user sessions represented in this spreadsheet: represented as one row for each user session. The first six columns measure the different types of pages users have visited in the session: the Administrative, Informational, and ProductRelated columns measure how many of those types of pages the user visited, and their corresponding _Duration columns measure how much time the user spent on any of those pages. The BounceRates, ExitRates, and PageValues columns measure information from Google Analytics about the page the user visited. SpecialDay is a value that measures how closer the date of the user’s session is to a special day (like Valentine’s Day or Mother’s Day). Month is an abbreviation of the month the user visited. OperatingSystems, Browser, Region, and TrafficType are all integers describing information about the user themself. VisitorType will take on the value Returning_Visitor for returning visitors and some other string value for non-returning visitors. Weekend is TRUE or FALSE depending on whether or not the user is visiting on a weekend.

Perhaps the most important column, though, is the last one: the Revenue column. This is the column that indicates whether the user ultimately made a purchase or not: TRUE if they did, FALSE if they didn’t. This is the column that we’d like to learn to predict (the “label”), based on the values for all of the other columns (the “evidence”).

Next, take a look at shopping.py. The main function loads data from a CSV spreadsheet by calling the load_data function and splits the data into a training and testing set. The train_model function is then called to train a machine learning model on the training data. Then, the model is used to make predictions on the testing data set. Finally, the evaluate function determines the sensitivity and specificity of the model, before the results are ultimately printed to the terminal.

There are two Python files in this project: `crossword.py` and `generate.py`. 

The first file defines two classes, `Variable` (to represent a variable in a crossword puzzle) and `Crossword` (to represent the puzzle itself). The `Crossword` class requires two values to create a new crossword puzzle: a `structure_file` that defines the structure of the puzzle and a `words_file` that defines a list of words to use for the vocabulary of the puzzle. Three examples of each of these files can be found in the `data` directory.

The `Crossword` object stores the `crossword.overlaps` which is a dictionary mapping a pair of variables to their overlap. For any two distinct variables it will be `None` if the two variables have no overlap, or will be a pair of integers `(i, j)` if the variables do overlap. The pair `(i, j)` should be interpreted to mean that the ith character of v1’s value must be the same as the jth character of v2’s value. The `Crossword` objects also support a method `neighbors` that returns all of the variables that overlap with a given variable.

In `generate.py`, we define a class `CrosswordCreator` that we’ll use to solve the crossword puzzle. When a `CrosswordCreator` object is created, it gets a `crossword` property that should be a `Crossword` object. Each `CrosswordCreator` object also gets a `domains` property: a dictionary that maps variables to a set of possible words the variable might take on as a value. Initially, this set of words is all of the words in our vocabulary. The solve function does three things: first, it calls `enforce_node_consistency` to enforce node consistency on the crossword puzzle, ensuring that every value in a variable’s domain satisfy the unary constraints. Next, the function calls `ac3` to enforce arc consistency, ensuring that binary constraints are satisfied. Finally, the function calls `backtrack` on an initially empty assignment to try to calculate a solution to the problem.

## Resources

* [Optimization - Lecture 3 - CS50's Introduction to Artificial Intelligence with Python 2020][cs50 lecture]

## Usage

**To install Scikit-learn:**

* Inside the `shopping` directory: `pip3 install scikit-learn`

**To train and test a model to predict online shopping purchases:** 

* Inside the `shopping` directory: `python shopping.py shopping.csv`

## Credits

[*Luis Sanchez*][linkedin] 2021.

Project and images from the course [CS50's Introduction to Artificial Intelligence with Python 2020][cs50 ai] from HarvardX.

[cs50 lecture]: https://youtu.be/qK46ET1xk2A?t=3070
[linkedin]: https://www.linkedin.com/in/luis-sanchez-13bb3b189/
[cs50 ai]: https://cs50.harvard.edu/ai/2020/



