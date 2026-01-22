import os
import random
import pickle
import face_recognition
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import google.genai as genai
from sklearn.linear_model import LinearRegression
from PIL import Image
import json
from google.genai import types, errors

instruction_path = "instructions.txt"

with open(instruction_path, "r", encoding="utf-8") as file:
    system_instructions = file.read()

api_key = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3-flash-preview"


def get_gemini_response(user_input, context):

    st.write("Thinking...")
    try:

        config = types.GenerateContentConfig(
            system_instruction=system_instructions, temperature=0.7)

        chat = client.chats.create(model=MODEL_NAME, config=config)

        full_message = f"{context}\n\nUser Question: {user_input}"

        response = chat.send_message(full_message)

        st.session_state.gemini_response = response.text

    except errors.ClientError as e:
        if e.code == 429:
            if "quota" in str(e).lower():
                st.session_state.gemini_response = "ERROR: Daily limit reached. Try again tomorrow."
            else:
                st.session_state.gemini_response = "ERROR: You're sending messages too fast! Wait 30 seconds."
        else:
            st.session_state.gemini_response = f"API Error: {str(e)}"

    except Exception as e:
        st.session_state.gemini_response = f"An unexpected error occurred: {str(e)}"


st.set_page_config(layout="wide")

DB_PATH = "bank_database.pkl"
INSTRUCTIONS_FILE = "instructions.txt"
ASSETS = "user_assets.pkl"

if os.path.exists(DB_PATH):
    with open(DB_PATH, "rb") as f:
        saved_faces = pickle.load(f)
else:
    saved_faces = {}

if os.path.exists(ASSETS):
    with open(ASSETS, "rb") as f:
        asset_data = pickle.load(f)
else:
    asset_data = {}


def pkl_to_json():
    with open(ASSETS, "rb") as f:
        asset_data = pickle.load(f)
    with open("assets.json", "w") as f:
        json.dump(asset_data, f, indent=4)


def init_assets(username):
    if username not in asset_data:

        days = list(range(1, 51))

        with open("stock_1.json", "r") as f:
            stock_data = json.load(f)
        stock_1_values = stock_data["prices"]

        with open("stock_2.json", "r") as f:
            stock_data = json.load(f)
        stock_2_values = stock_data["prices"]

        with open("stock_3.json", "r") as f:
            stock_data = json.load(f)
        stock_3_values = stock_data["prices"]

        with open("stock_4.json", "r") as f:
            stock_data = json.load(f)
        stock_4_values = stock_data["prices"]

        asset_data[username] = {
            "total_money": 10000,
            "total_stock_1": 0,
            "stock_1_value": 0,
            "total_stock_2": 0,
            "stock_2_value": 0,
            "total_stock_3": 0,
            "stock_3_value": 0,
            "total_stock_4": 0,
            "stock_4_value": 0,
            "stock_1_days": days,
            "stock_2_days": days,
            "stock_3_days": days,
            "stock_4_days": days,
            "stock_1_values": stock_1_values,
            "stock_2_values": stock_2_values,
            "stock_3_values": stock_3_values,
            "stock_4_values": stock_4_values
        }

        with open(ASSETS, "wb") as f:
            pickle.dump(asset_data, f)
        pkl_to_json()
    else:
        pkl_to_json()


def update_assets(key, num):
    if st.session_state.username in asset_data:
        asset_data[st.session_state.username][key] = num
        with open(ASSETS, "wb") as f:
            pickle.dump(asset_data, f)
        pkl_to_json()


def get_user_asset(key):
    if st.session_state.username in asset_data:
        return asset_data[st.session_state.username].get(key)
    return None


def reset_assets():

    days = list(range(1, 51))
    with open("stock_1.json", "r") as f:
        stock_data = json.load(f)
    stock_1_values = stock_data["prices"][:50]

    with open("stock_2.json", "r") as f:
        stock_data = json.load(f)
    stock_2_values = stock_data["prices"][:50]

    with open("stock_3.json", "r") as f:
        stock_data = json.load(f)
    stock_3_values = stock_data["prices"][:50]

    with open("stock_4.json", "r") as f:
        stock_data = json.load(f)
    stock_4_values = stock_data["prices"][:50]

    asset_data[st.session_state.username] = {
        "total_money": 10000,
        "total_stock_1": 0,
        "stock_1_value": 0,
        "total_stock_2": 0,
        "stock_2_value": 0,
        "total_stock_3": 0,
        "stock_3_value": 0,
        "total_stock_4": 0,
        "stock_4_value": 0,
        "stock_1_days": days,
        "stock_2_days": days,
        "stock_3_days": days,
        "stock_4_days": days,
        "stock_1_values": stock_1_values,
        "stock_2_values": stock_2_values,
        "stock_3_values": stock_3_values,
        "stock_4_values": stock_4_values
    }

    with open(ASSETS, "wb") as f:
        pickle.dump(asset_data, f)

    pkl_to_json()

    st.rerun()


st.title("BANK OF CARSON")


def reset():
    st.session_state.username = ""
    st.session_state.is_verified = False

    st.session_state.username = ""
    st.session_state.is_verified = False


def check_bankruptcy():
    money = get_user_asset("total_money")
    s1 = get_user_asset("stock_1_value")
    s2 = get_user_asset("stock_2_value")
    s3 = get_user_asset("stock_3_value")
    s4 = get_user_asset("stock_4_value")

    if money is None or s1 is None or s2 is None or s3 is None or s4 is None:
        return

    net_worth = money + s1 + s2 + s3 + s4

    if net_worth <= 0:
        st.error(
            "BANKRUPTCY DETECTED! Your net worth has hit $0. Resetting account..."
        )
        reset_assets()
        st.session_state.step = "bank_account"
        st.rerun()


if "step" not in st.session_state:
    st.session_state.step = "menu"
if "username" not in st.session_state:
    st.session_state.username = ""
if "is_verified" not in st.session_state:
    st.session_state.is_verified = False
if "gemini_response" not in st.session_state:
    st.session_state.gemini_response = ""

with st.sidebar:
    if st.button("🏠 Home"):
        st.session_state.username = ""
        st.session_state.is_verified = False
        st.session_state.step = "menu"
        st.rerun()
    if st.session_state.is_verified:
        if st.button("💰 Bank Account"):
            st.session_state.step = "bank_account"
            st.rerun()
        if st.button("📈 Stock Market"):
            st.session_state.step = "stock_market"
            st.rerun()
        if st.button("🏛️ Your assets"):
            st.session_state.step = "assets"
            st.rerun()
        if st.button("📊 Compare Stocks"):
            st.session_state.step = "compare_stocks"
            st.rerun()

if st.session_state.step == "menu":
    if st.button("Access your account"):
        st.session_state.step = "login_name"
        st.rerun()
    if st.button("Create new account"):
        st.session_state.step = "signup_name"
        st.rerun()

elif st.session_state.step == "signup_name":
    name = st.text_input("Choose a username and press Enter")
    if name:
        if name in saved_faces:
            st.error(
                f"The username '{name}' is already taken. Please choose another."
            )
        else:

            st.session_state.username = name
            st.session_state.step = "signup_camera"
            st.rerun()

elif st.session_state.step == "signup_camera":
    st.subheader(f"Registering face for: {st.session_state.username}")
    img_file = st.camera_input("Take a photo")
    if img_file:
        img = Image.open(img_file)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_array = np.array(img, dtype=np.uint8)

        img_array = np.ascontiguousarray(img_array)

        encodings = face_recognition.face_encodings(img_array)
        if encodings:
            saved_faces[st.session_state.username] = encodings[0]
            with open(DB_PATH, "wb") as f:
                pickle.dump(saved_faces, f)

            st.success(
                "Your username and face are now registered! To access your new bank account, return back to the main menu and press the 'Access your account' button"
            )
        else:
            st.error("No face detected, try again.")

elif st.session_state.step == "login_name":
    name = st.text_input("Enter your username and press Enter")
    if name:
        if name in saved_faces:
            st.session_state.username = name
            st.session_state.step = "login_camera"

            st.rerun()
        else:
            st.error("Username not found.")

elif st.session_state.step == "login_camera":
    st.subheader(f"Verifying: {st.session_state.username}")
    img_file = st.camera_input("Verify Identity")
    if img_file:
        img = Image.open(img_file)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_array = np.array(img, dtype=np.uint8)

        img_array = np.ascontiguousarray(img_array)

        login_encoding = face_recognition.face_encodings(img_array)
        if login_encoding:

            match = face_recognition.compare_faces(
                [saved_faces[st.session_state.username]], login_encoding[0])
            if match[0]:
                st.success("Verified!")
                st.session_state.step = "bank_account"
                init_assets(st.session_state.username)
                st.session_state.is_verified = True
                st.rerun()
            else:
                st.error("Face does not match.")
        else:
            st.error("No face detected.")

elif st.session_state.step == "bank_account":
    st.subheader(f"Welcome, {st.session_state.username}")
    if st.button("Stock market"):
        st.session_state.step = "stock_market"
        st.rerun()

    if st.button("Your assets"):
        st.session_state.step = "assets"
        st.rerun()
    if st.button("Compare Stocks"):
        st.session_state.step = "compare_stocks"
        st.rerun()

elif st.session_state.step == "stock_market":
    st.subheader("Stock Market Simulation")
    check_bankruptcy()

    stock_options = [
        "Dave and Son's Coal Mine", "Xavier's Egg Farm",
        "Mr. Fox's Chicken Company", "Raymond's Water Company"
    ]

    display_name = st.selectbox("Select a stock to analyse", stock_options)

    if display_name == stock_options[0]:
        index = 0
        asset_value = "stock_1_values"
        asset_number = "total_stock_1"
        asset_value_key = "stock_1_value"
        asset_days = "stock_1_days"

    elif display_name == stock_options[1]:
        index = 1
        asset_value = "stock_2_values"
        asset_number = "total_stock_2"
        asset_value_key = "stock_2_value"
        asset_days = "stock_2_days"

    elif display_name == stock_options[2]:
        index = 2
        asset_value = "stock_3_values"
        asset_number = "total_stock_3"
        asset_value_key = "stock_3_value"
        asset_days = "stock_3_days"

    elif display_name == stock_options[3]:
        index = 3
        asset_value = "stock_4_values"
        asset_number = "total_stock_4"
        asset_value_key = "stock_4_value"
        asset_days = "stock_4_days"
    else:
        index = 0
        asset_value = "stock_1_values"
        asset_number = "total_stock_1"
        asset_value_key = "stock_1_value"
        asset_days = "stock_1_days"

    graph, panel = st.columns([3, 1])

    with panel:
        target_days = st.text_input("Simulate to day (extrapolation)",
                                    placeholder="Enter day number",
                                    key="sim_input")

        if st.button("Run Simulation"):
            if target_days:
                try:
                    target_days_int = int(target_days)

                    for stock_num in range(1, 5):
                        stock_days_key = f"stock_{stock_num}_days"
                        stock_values_key = f"stock_{stock_num}_values"

                        current_days = get_user_asset(stock_days_key)
                        current_values = get_user_asset(stock_values_key)

                        if current_days is None or current_values is None:
                            continue

                        current_days = list(current_days)
                        current_values = list(current_values)
                        current_max_day = max(current_days)

                        if target_days_int > current_max_day:
                            x = np.array(current_days).reshape(-1, 1)
                            y = np.array(current_values)
                            model = LinearRegression()
                            model.fit(x, y)

                            new_days = []
                            new_prices = []

                            for i in range(current_max_day + 1,
                                           target_days_int + 1):
                                prediction = model.predict([[i]])[0]

                                n_min = random.uniform(-0.03, -0.01)
                                n_max = random.uniform(0.02, 0.05)
                                noise = prediction * np.random.uniform(
                                    n_min, n_max)
                                new_price = round(float(prediction + noise), 2)

                                new_days.append(i)

                                if new_price < 1:
                                    new_prices.append(1)
                                else:
                                    new_prices.append(new_price)

                            current_days.extend(new_days)
                            current_values.extend(new_prices)

                            update_assets(stock_days_key, current_days)
                            update_assets(stock_values_key, current_values)

                    check_bankruptcy()
                    st.success(
                        f"Simulated ALL stocks up to day {target_days_int}")
                    st.rerun()

                except ValueError:
                    st.error("Please enter a valid number")

    days = get_user_asset(asset_days)
    value = get_user_asset(asset_value)

    if days is None or value is None:
        st.error(
            "Error loading stock data. Please try logging in again or reset your assets."
        )
        if st.button("Reset Assets"):
            reset_assets()
        st.stop()

    days = list(days)
    value = list(value)

    min_len = min(len(days), len(value))
    days = days[:min_len]
    value = value[:min_len]

    fig, ax = plt.subplots()

    ax.scatter(days, value)
    ax.set_title(display_name)

    x_min, x_max = ax.get_xlim()

    m, b = np.polyfit(days, value, 1)

    if m > 0:
        line_color = "#32CD32"
    else:
        line_color = "#EE4B2B"

    y_min = m * x_min + b
    y_max = m * x_max + b

    ax.plot([x_min, x_max], [y_min, y_max],
            color=line_color,
            linewidth=2.5,
            label='Line of best fit')

    ax.set_xlim(x_min, x_max)

    ax.set_xlabel("Days")
    ax.set_ylabel("Stock price ($)")

    with graph:
        st.pyplot(fig)
        st.divider()
        st.subheader("AI Investment Advisor")

        user_question = st.text_input(
            "Ask about investment tips, which stocks to invest in, and any questions related to the ISU"
        )

        if st.button("Submit") and user_question:
            try:

                user_assets = {}

                with open("assets.json", "r") as f:
                    user_assets = json.load(f)

                context = f"ALL DATA, USER AND STOCK: {user_assets}\n\n"
                context += f"Current Money: {get_user_asset('total_money')}\n"
                context += f"All Stock Holdings: Stock1={get_user_asset('total_stock_1')}, Stock2={get_user_asset('total_stock_2')}, Stock3={get_user_asset('total_stock_3')}, Stock4={get_user_asset('total_stock_4')}\n\n"

                get_gemini_response(user_question, context)

            except Exception as e:
                st.session_state.gemini_response = f"Error getting AI advice: {e}"

        if st.session_state.gemini_response:
            st.write("AI Response:")
            st.write(st.session_state.gemini_response)

        if st.button("Clear AI Response"):
            st.session_state.gemini_response = ""
            st.rerun()
    st.divider()

    if st.button("Reset All"):
        reset_assets()

    with panel:
        current_money = get_user_asset("total_money")
        current_value_list = get_user_asset(asset_value)

        if current_money is None or current_value_list is None:
            st.error("Error loading data")
            st.stop()

        current_price = current_value_list[-1]

        st.divider()
        buy_input = st.number_input("Buy shares", min_value=0, step=1)

        if buy_input > 0:
            stock_number = get_user_asset(asset_number)

            if stock_number is None:
                st.error("Error loading stock data")
            else:
                total_cost = np.abs(current_price * buy_input)
                balance_after = current_money - total_cost
                shares_after = stock_number + buy_input

                st.write(f"Current Shares: {stock_number:,.2f}")
                st.write(f"Shares After: {shares_after:,.2f}")
                st.write(f"Current Balance: ${current_money:,.2f}")
                st.write(f"Cost: ${total_cost:,.2f}")
                st.write(f"Balance After: ${balance_after:,.2f}")

                if total_cost > current_money:
                    st.error("Insufficient funds!")
                else:
                    if st.button("Confirm Purchase"):

                        update_assets("total_money", balance_after)
                        update_assets(asset_number, shares_after)
                        update_assets(asset_value_key,
                                      shares_after * current_price)

                        st.success(f"Bought {buy_input} shares!")

                        st.rerun()

        st.divider()
        sell_input = st.number_input("Sell shares", min_value=0, step=1)

        if sell_input > 0:
            stock_number = get_user_asset(asset_number)

            if stock_number is None:
                st.error("Error loading stock data")
            elif sell_input > stock_number:
                st.error("Not enough shares!")
            else:
                gain = np.abs(sell_input * current_price)
                balance_after = current_money + gain
                shares_after = stock_number - sell_input

                st.write(f"Current Shares: {stock_number:,.2f}")
                st.write(f"Shares After: {shares_after:,.2f}")
                st.write(f"Current Balance: ${current_money:,.2f}")
                st.write(f"Earnings: ${gain:,.2f}")
                st.write(f"Balance After: ${balance_after:,.2f}")

                if st.button("Confirm Sale"):
                    update_assets("total_money", balance_after)
                    update_assets(asset_number, shares_after)
                    update_assets(asset_value_key,
                                  shares_after * current_price)

                    st.success(f"Sold {sell_input} shares!")
                    st.rerun()

elif st.session_state.step == "assets":
    st.divider()

    st.subheader(f"Portfolio for {st.session_state.username}")

    account_balance = get_user_asset("total_money")
    s1_qty = get_user_asset("total_stock_1")
    s1_val = get_user_asset("stock_1_value")
    s2_qty = get_user_asset("total_stock_2")
    s2_val = get_user_asset("stock_2_value")
    s3_qty = get_user_asset("total_stock_3")
    s3_val = get_user_asset("stock_3_value")
    s4_qty = get_user_asset("total_stock_4")
    s4_val = get_user_asset("stock_4_value")

    if any(x is None for x in [
            account_balance, s1_qty, s1_val, s2_qty, s2_val, s3_qty, s3_val,
            s4_qty, s4_val
    ]):
        st.error("Error loading portfolio data")
        st.stop()

    net_worth = account_balance + s1_val + s2_val + s3_val + s4_val

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Net Worth", f"${net_worth:,.2f}")
        st.metric("Account Balance", f"${account_balance:,.2f}")

    with col2:
        st.write("Stock Holdings")
        asset_table = {
            "Stock Name": [
                "Dave and Son's Coal Mine", "Xavier's Egg Farm",
                "Mr. Fox's Chicken Company", "Raymond's Water Company"
            ],
            "Shares Owned": [s1_qty, s2_qty, s3_qty, s4_qty],
            "Current Value": [
                f"${s1_val:,.2f}", f"${s2_val:,.2f}", f"${s3_val:,.2f}",
                f"${s4_val:,.2f}"
            ]
        }
        st.table(asset_table)
    st.divider()
    if st.button("Back to Bank Account"):
        st.session_state.step = "bank_account"
        st.rerun()

elif st.session_state.step == "compare_stocks":
    st.divider()
    st.subheader("Compare Stocks")

    view_mode = st.selectbox("View mode: ", ["Histogram", "Box Plot"])

    names = [
        "Dave and Son's Coal Mine", "Xavier's Egg Farm",
        "Mr. Fox's Chicken Company", "Raymond's Water Company"
    ]

    p1 = get_user_asset("stock_1_values")
    p2 = get_user_asset("stock_2_values")
    p3 = get_user_asset("stock_3_values")
    p4 = get_user_asset("stock_4_values")

    if any(x is None for x in [p1, p2, p3, p4]):
        st.error("Error loading stock data")
        st.stop()

    all_prices = [p1, p2, p3, p4]

    if view_mode == "Histogram":
        fig, ax = plt.subplots(figsize=(12, 6))

        max_len = max(len(p) for p in all_prices)
        step = max(1, max_len // 10)
        indices = list(range(0, max_len, step))
        if (max_len - 1) not in indices:
            indices.append(max_len - 1)

        x = np.arange(len(indices))
        width = 0.2

        ax.bar(x, [p1[min(i,
                          len(p1) - 1)] for i in indices],
               width,
               label=names[0])
        ax.bar(x + width, [p2[min(i,
                                  len(p2) - 1)] for i in indices],
               width,
               label=names[1])
        ax.bar(x + width * 2, [p3[min(i,
                                      len(p3) - 1)] for i in indices],
               width,
               label=names[2])
        ax.bar(x + width * 3, [p4[min(i,
                                      len(p4) - 1)] for i in indices],
               width,
               label=names[3])

        ax.set_xlabel("Time (Day)")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([f"Day {i+1}" for i in indices])
        ax.set_ylabel("Price ($)")
        ax.legend()
        st.pyplot(fig)

    elif view_mode == "Box Plot":
        graph, stats = st.columns([3.1, 2])

        with graph:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(all_prices, tick_labels=names)
            ax.set_ylabel("Stock Price ($)")
            plt.xticks(rotation=15)
            st.pyplot(fig)

        with stats:
            st.subheader("Statistics")

            def get_stats(p_list, name):
                arr = np.array(p_list)
                q1 = np.percentile(arr, 25)
                q3 = np.percentile(arr, 75)
                return [
                    name, f"${np.median(arr):.2f}", f"${q1:.2f}", f"${q3:.2f}",
                    f"${q3-q1:.2f}"
                ]

            rows = [
                get_stats(p1, names[0]),
                get_stats(p2, names[1]),
                get_stats(p3, names[2]),
                get_stats(p4, names[3])
            ]

            stat_table = {
                "Stock": [r[0] for r in rows],
                "Median": [r[1] for r in rows],
                "Q1": [r[2] for r in rows],
                "Q3": [r[3] for r in rows],
                "IQR": [r[4] for r in rows]
            }
            st.table(stat_table)

    st.divider()
    if st.button("Back to Bank Account"):
        st.session_state.step = "bank_account"
        st.rerun()
