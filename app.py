import streamlit as st
import pandas as pd
import requests
from joblib import load, dump
from sklearn.linear_model import LogisticRegression
import logging





def predict(model,preprocessor,data_df):
    x_transformed = preprocessor.transform(data_df)
    prediction = model.predict(x_transformed)
    return prediction[0]


def main():
    st.title("Recipe Review Sentiment Analysis")

    recipe_names = ['Banana Bars with Cream Cheese Frosting', 'Simple Taco Soup',
                    'Cheeseburger Soup', 'Grilled Huli Huli Chicken',
                    'Cauliflower Soup', 'Favorite Chicken Potpie',
                    'Fluffy Key Lime Pie', 'Seafood Lasagna', 'Cheesy Ham Chowder',
                    'First-Place Coconut Macaroons', 'Rhubarb Custard Bars',
                    'Garlic Beef Enchiladas', 'Pork Chops with Scalloped Potatoes',
                    'Peanut Butter Cup Cheesecake', 'Li’l Cheddar Meat Loaves',
                    'Chicken Wild Rice Soup', 'Flavorful Chicken Fajitas',
                    'Enchilada Casser-Ole!', 'Creamy White Chili',
                    'Stuffed Pepper Soup', 'Lemon Blueberry Bread',
                    'Chicken and Dumplings', 'Porcupine Meatballs',
                    'Simple Au Gratin Potatoes', 'Basic Homemade Bread',
                    'Zucchini Cupcake', 'Forgotten Jambalaya', 'Ravioli Lasagna',
                    'Contest-Winning New England Clam Chowder', 'Basic Banana Muffins',
                    'Creamy Grape Salad', 'Big Soft Ginger Cookies',
                    'Pumpkin Spice Cupcakes with Cream Cheese Frosting',
                    'Shrimp Scampi', 'Chocolate-Strawberry Celebration Cake',
                    'Easy Peanut Butter Fudge', 'Brown Sugar Oatmeal Pancakes',
                    'Winning Apple Crisp', 'White Bean Chicken Chili',
                    'Creamy Coleslaw', 'Strawberry Pretzel Salad', 'Hot Milk Cake',
                    'Caramel Heavenlies', 'Peanut Butter Chocolate Dessert',
                    'Teriyaki Chicken Thighs', 'Cheeseburger Paradise Soup',
                    'Apple Pie', 'Favorite Dutch Apple Pie',
                    'Amish Breakfast Casserole', 'Easy Chicken Enchiladas',
                    'Pumpkin Bread', 'Pineapple Orange Cake', 'Slow-Cooker Lasagna',
                    'Baked Tilapia', 'Zucchini Pizza Casserole',
                    'Best Ever Banana Bread', 'Corn Pudding',
                    'Chicken Penne Casserole', 'Gluten-Free Banana Bread',
                    'Chocolate Chip Oatmeal Cookies', 'Comforting Chicken Noodle Soup',
                    'Sandy’s Chocolate Cake', 'Bruschetta Chicken',
                    'Rustic Italian Tortellini Soup', 'Black Bean ‘n’ Pumpkin Chili',
                    'Lime Chicken Tacos', 'Best Ever Potato Soup', 'Buttery Cornbread',
                    'Special Banana Nut Bread', 'Frosted Banana Bars',
                    'Smothered Chicken Breasts', 'Quick Cream of Mushroom Soup',
                    'Vegetarian Linguine', 'Asian Chicken Thighs',
                    'Traditional Lasagna', 'Mom’s Meat Loaf', 'Flavorful Pot Roast',
                    'Caramel-Pecan Cheesecake Pie', 'Tennessee Peach Pudding',
                    'Creamy Macaroni and Cheese', 'Homemade Peanut Butter Cups',
                    'Chunky Apple Cake', 'Skillet Shepherd’s Pie',
                    'Egg Roll Noodle Bowl', 'Cherry Bars',
                    'Mamaw Emily’s Strawberry Cake', 'Pineapple Pudding Cake',
                    'Moist Chocolate Cake', 'Macaroni Coleslaw', 'Pumpkin Bars',
                    'Baked Spaghetti', 'Chocolate Guinness Cake',
                    'Ham and Swiss Sliders', 'Twice-Baked Potato Casserole',
                    'Taco Lasagna', 'Bacon Macaroni Salad', 'Chocolate Caramel Candy',
                    'Baked Mushroom Chicken', 'Fluffy Pancakes',
                    'Blueberry French Toast']
    
    recipe_name = st.selectbox('Select Recipe', recipe_names)
    user_name = st.text_input('Username')
    recipe_review = st.text_area('write Recipe Review here...')
    id = st.number_input('ID')
    user_repo = st.number_input("UserReputation")
    reply_count = st.number_input("ReplyCount")
    thumbs_up_count = st.number_input("ThumbsUpCount")
    thumbs_down_count = st.number_input("ThumbsDownCount")
    best_score = st.number_input("BestScore")
    date = st.date_input("Enter date of review...",value=None)
    hour = st.number_input("Hour of the Recipe Review.")

    if st.button('Predict'):
        review_length = len(recipe_review)
        year = date.year
        month = date.month
        dayofweek = date.weekday()
        columns = ['ID','UserReputation', 'ReplyCount', 'ThumbsUpCount', 'ThumbsDownCount', 'BestScore', 'Review_length',
                    'year', 'month', 'dayofweek','hour','RecipeName','UserName','Recipe_Review']
        data = [[id,user_repo,reply_count,thumbs_up_count,thumbs_down_count,best_score,review_length,year,month,dayofweek
                ,hour,recipe_name,user_name,recipe_review]]
        data_df = pd.DataFrame(data=data,columns=columns)
        model = load(r'models\best_model.joblib')
        preprocessor = load(r'preprocessors\preprocessor.joblib')
        if model is not None:
            with st.spinner('Predicting...'):
                prediction = predict(model,preprocessor,data_df)
            st.markdown(f'<h1 style="font-size:48px; color:green">Predicted Rating of Recipe is: {prediction}</h1>', unsafe_allow_html=True)
if __name__ == '__main__':
    main()