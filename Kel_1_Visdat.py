# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
# from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
# pd.set_option('mode.use_inf_as_null', True)

color = ['#E966A0','#2B2730','#6554AF','#9575DE','#0E2954','#1F6E8C','#2E8A99','#84A7A1']

# + colab={"base_uri": "https://localhost:8080/", "height": 427} id="ohigkTo-sQ9v" outputId="b68455c4-6c33-42af-ba13-e26c8f16a443"
dataset1 = pd.read_csv('Hotel Reservations.csv')
dataset1.head(10)

# + colab={"base_uri": "https://localhost:8080/", "height": 427} id="bujyUQd-t31a" outputId="ea85b909-e100-4313-e964-5b266072d291"
dataset1.tail(10)

# + colab={"base_uri": "https://localhost:8080/"} id="zsbuz5iyuY9_" outputId="fbda3663-1a31-4648-8fee-517d67c95e72"
dataset1.info()

# + colab={"base_uri": "https://localhost:8080/"} id="73ADmMh2vTFr" outputId="a5d513ba-162c-4f82-f2fe-f39f7a266598"
dataset1['booking_status'].value_counts()

# + colab={"base_uri": "https://localhost:8080/"} id="m6DVCWji1bI_" outputId="557eae34-240e-40d1-9737-f5912c5f8722"
dataset1['arrival_year'].value_counts()
# -

dataset2 = dataset1[dataset1['booking_status']=='Not_Canceled'] # melakukan pengelompokkan berdasarkan keuntungan (booking not cancelled)
dataset2

# + [markdown] id="A2F0pom6wGAl"
# # Pemilik Hotel

# + colab={"base_uri": "https://localhost:8080/", "height": 270} id="6g6VmvSrv4K7" outputId="9744ac9a-dbbe-4628-e3ed-6a4ff6cd36a3"
pemilik_hotel = dataset2.drop(['repeated_guest',	'no_of_previous_cancellations',	'no_of_previous_bookings_not_canceled', 'required_car_parking_space', 'lead_time', 'no_of_special_requests'], axis = 'columns')
# pemilik_hotel = pemilik_hotel[pemilik_hotel['booking_status']=='Not_Canceled']
pemilik_hotel.head()
# -

# Booking_status jangan di drop karena bisa dikelompokan jumlah jadi dan tidak jadi 

# + colab={"base_uri": "https://localhost:8080/", "height": 237} id="KrlQA6seCqy7" outputId="ec074755-cb41-4cf3-95d6-c6de39d25ed2"
pemilik_hotel_sortBy_roomType = pemilik_hotel.drop(['Booking_ID','no_of_adults',	'no_of_children',	'no_of_weekend_nights',	'no_of_week_nights',
                                                    'type_of_meal_plan', 'arrival_year',	'arrival_month',	'arrival_date',	'market_segment_type'], axis = 'columns').groupby(by='room_type_reserved').sum()
pemilik_hotel_sortBy_roomType.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 472} id="26fl52Z_LU9C" outputId="600ac899-4a9e-41f5-ce0d-28a8bfe6f326"
x = list(pemilik_hotel_sortBy_roomType.index)
y = pemilik_hotel_sortBy_roomType['avg_price_per_room']
fig = plt.figure(figsize = (19, 10))
plt.bar(x, y, color = color,width = 0.7)
plt.xlabel('X Label', fontsize=12)
plt.ylabel('Y Label', fontsize=12)
plt.xlabel("Room Type")
plt.ylabel("Sum of Price (*10^6)")
plt.title("Grafik keuntungan setiap tipe kamar", fontsize=16, fontweight='bold', y=1.01)
plt.show()
st.plotly(fig)
# -

pemilik_hotel_year2017 = pemilik_hotel[pemilik_hotel['arrival_year'] == 2017]
pemilik_hotel_year2017

pemilik_hotel_groupBy_Month2017 = pemilik_hotel_year2017[['arrival_month', 'avg_price_per_room']].groupby(by='arrival_month').sum()
pemilik_hotel_groupBy_Month2017

x = list(pemilik_hotel_groupBy_Month2017.index)
y = pemilik_hotel_groupBy_Month2017['avg_price_per_room']
fig = plt.figure(figsize = (15, 9))
plt.plot(x, y,)
plt.xlabel("Bulan")
plt.ylabel("Jumlah")
plt.title("Grafik keuntungan setiap bulannya pada tahun 2017", fontsize=16, fontweight='bold', y=1.01)
plt.show()
st.plotly(fig)

pemilik_hotel_year2018 = pemilik_hotel[pemilik_hotel['arrival_year'] == 2018]
pemilik_hotel_groupBy_Month2018 = pemilik_hotel_year2018[['arrival_month', 'avg_price_per_room']].groupby(by='arrival_month').sum()
pemilik_hotel_groupBy_Month2018

x = list(pemilik_hotel_groupBy_Month2018.index)
y = pemilik_hotel_groupBy_Month2018['avg_price_per_room']
fig = plt.figure(figsize = (15, 7))
plt.plot(x, y)
plt.xlabel("Bulan")
plt.ylabel("Jumlah")
plt.title("Grafik keuntungan setiap bulannya pada tahun 2018", fontsize=16, fontweight='bold', y=1.01)
plt.show()
st.plotly(fig)

# +
colunas = ['arrival_year','arrival_month','arrival_date']

plt.figure(figsize=(14,10))
for i, col in enumerate(colunas):
    ax = plt.subplot(3,3,i+1)
    sns.lineplot(x=dataset2[col],y=dataset2['avg_price_per_room'])
    plt.title(col, fontweight='bold')

plt.suptitle("Rentang Keuntungan Perusahaan berdasarkan tahun, bulan, dan Hari", fontsize=16, fontweight='bold', x=0.50,y=1.00)
plt.tight_layout()
plt.show()
st.plotly(colunas)

# +
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 2, figsize=(14, 4))

sns.lineplot(x=dataset2['no_of_week_nights'], y=dataset2['avg_price_per_room'], ax=ax[0])
sns.lineplot(x=dataset2['no_of_weekend_nights'], y=dataset2['avg_price_per_room'], ax=ax[1])

plt.suptitle("Rentang Keuntungan Perusahaan berdasarkan akhir pekan dan hari pekan", fontsize=16, fontweight='bold', x=0.50)
plt.show()
st.plotly(fig)

# + colab={"base_uri": "https://localhost:8080/", "height": 143} id="PNHwnmtchw8S" outputId="4dbf35a1-8f41-4edf-c573-a05f20ba6c69"
pemilik_hotel_sortBy_arrivalYear = pemilik_hotel.drop(['Booking_ID',	'no_of_adults',	'no_of_children',	'no_of_weekend_nights',	'no_of_week_nights',
                                                    'type_of_meal_plan', 'room_type_reserved',	'arrival_month',	'arrival_date',	'market_segment_type'], axis = 'columns').groupby(by='arrival_year').sum()
total_avg_price_per_room = pemilik_hotel_sortBy_arrivalYear['avg_price_per_room'].sum()

# Create a new DataFrame with the total row
total_row = pd.DataFrame({'avg_price_per_room': [total_avg_price_per_room]}, index=['Total'])

# Append the total row to the existing DataFrame
# pemilik_hotel_sortBy_arrivalYear_with_total = pemilik_hotel_sortBy_arrivalYear.append(total_row, ignore_index=True)
pemilik_hotel_sortBy_arrivalYear_with_total = pd.concat([pemilik_hotel_sortBy_arrivalYear, total_row], ignore_index=True)

pemilik_hotel_sortBy_arrivalYear_with_total.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 428} id="vCv6Yz_lpDk_" outputId="42025c25-55a1-4144-c6f8-9d4850c8d126"
import matplotlib.pyplot as plt

kategori2 = list(pemilik_hotel_sortBy_arrivalYear.index)
nilai2 = pemilik_hotel_sortBy_arrivalYear['avg_price_per_room']

fig, ax = plt.subplots()
potongan, label, _ = ax.pie(nilai2, labels=kategori2, autopct='%1.1f%%')

ax.set_title('Rentang keuntungan Hotel berdasarkan Tahun', fontsize=16, fontweight='bold', x=0.50)
ax.legend(potongan, kategori2, loc='lower left', bbox_to_anchor=(1.1, 0.5))

plt.show()
st.plotly(fig)

# +
import matplotlib.pyplot as plt
import seaborn as sns

colunas = ['no_of_week_nights', 'no_of_weekend_nights']

plt.figure(figsize=(16, 10))

for i, col in enumerate(colunas):
    ax = plt.subplot(3, 3, i + 1)
    sns.barplot(x=dataset2[col], y=dataset2['avg_price_per_room'], palette=color)
    plt.title(col, fontweight='bold')

plt.suptitle("Rentang Keuntungan Perusahaan berdasarkan malam hari pekan dan malam akhir pekan", fontsize=16, fontweight='bold', x=0.35)
plt.tight_layout()
plt.show()
st.plotly(fig)

# + [markdown] id="-Zh3Dkd3pawY"
# # Front Officer
# -

front_officer = dataset2.drop(['market_segment_type','avg_price_per_room','no_of_adults','no_of_children','no_of_weekend_nights','no_of_week_nights','repeated_guest','no_of_previous_cancellations','no_of_previous_bookings_not_canceled', 'required_car_parking_space', 'lead_time', 'no_of_special_requests', 'booking_status'], axis = 'columns')
front_officer

# le = LabelEncoder()
front_officer['arrival_year'].value_counts()

front_officer_sortRoomType_byBookingID = front_officer.drop(['arrival_year', 
                                                               'arrival_month',
                                                               'arrival_date', 'type_of_meal_plan'], axis = 'columns').groupby(by='room_type_reserved').count()

front_officer_sortRoomType_byBookingID

front_officer_year2018 = front_officer[front_officer['arrival_year'] == 2018]
front_officer_year2018

front_officer_year2018['arrival_year'].value_counts()

front_officer_year2018_groupBy_roomType = front_officer_year2018[['Booking_ID', 'room_type_reserved']].groupby(by='room_type_reserved').count()
front_officer_year2018_groupBy_roomType

x = list(front_officer_year2018_groupBy_roomType.index)
y = front_officer_year2018_groupBy_roomType['Booking_ID']
fig = plt.figure(figsize = (10, 6))
plt.bar(x, y, color = color,width = 0.7)
plt.xlabel('X Label', fontsize=12)
plt.ylabel('Y Label', fontsize=12)
plt.xlabel("Room Type")
plt.ylabel("Total Room")
plt.title("Grafik pemesanan setiap tipe kamar terlaris per 2018", fontsize=16, fontweight='bold', y=1.01)
plt.show()
st.plotly(fig)

front_officer_year2017 = front_officer[front_officer['arrival_year'] == 2017]
front_officer_year2017

front_officer_year2017_groupBy_roomType = front_officer_year2017[['Booking_ID', 'room_type_reserved']].groupby(by='room_type_reserved').count()
front_officer_year2017_groupBy_roomType

x = list(front_officer_year2017_groupBy_roomType.index)
y = front_officer_year2017_groupBy_roomType['Booking_ID']
fig = plt.figure(figsize = (10, 6))
plt.bar(x, y, color = color,width = 0.7)
plt.xlabel('X Label', fontsize=12)
plt.ylabel('Y Label', fontsize=12)
plt.xlabel("Room Type")
plt.ylabel("Total Room")
plt.title("Grafik pemesanan setiap tipe kamar terlaris per 2017", fontsize=16, fontweight='bold', y=1.01)
plt.show()

overall_room_type_reserved = front_officer[['room_type_reserved', 'Booking_ID']].groupby(by='room_type_reserved').count()
overall_room_type_reserved

overall_type_of_meal_plan = front_officer[['type_of_meal_plan', 'Booking_ID']].groupby(by='type_of_meal_plan').count()
overall_type_of_meal_plan
st.plotly(fig)

# +
# Calculate the total number of booking IDs
total_booking_ids = overall_room_type_reserved['Booking_ID'].sum()

# Create a new DataFrame with the total row
total_row = pd.DataFrame({'Booking_ID': [total_booking_ids]}, index=['Total'])

# Append the total row to the existing DataFrame
# overall_room_type_reserved_with_total = overall_room_type_reserved.append(total_row)
overall_room_type_reserved_with_total = pd.concat([overall_room_type_reserved, total_row], ignore_index=True)

overall_room_type_reserved_with_total

# +
# Calculate the total number of booking IDs
total_booking_idss = overall_room_type_reserved['Booking_ID'].sum()

# Create a new DataFrame with the total row
total_row = pd.DataFrame({'Booking_ID': [total_booking_idss]}, index=['Total'])

# Append the total row to the existing DataFrame
# overall_type_of_meal_plan_with_total = overall_type_of_meal_plan.append(total_row)
overall_type_of_meal_plan_with_total = pd.concat([overall_type_of_meal_plan, total_row], ignore_index=True)

overall_type_of_meal_plan_with_total
# -

x = list(overall_room_type_reserved.index)
y = overall_room_type_reserved['Booking_ID']
fig = plt.figure(figsize = (10, 6))
plt.bar(x, y, color = color,width = 0.7)
plt.xlabel('X Label', fontsize=12)
plt.ylabel('Y Label', fontsize=12)
plt.xlabel("Room Type")
plt.ylabel("Total Room")
plt.title("Grafik pemesanan setiap tipe kamar terlaris", fontsize=16, fontweight='bold', y=1.01)
plt.show()
st.plotly(fig)

x = list(overall_type_of_meal_plan.index)
y = overall_type_of_meal_plan['Booking_ID']
fig = plt.figure(figsize = (10, 6))
plt.bar(x, y, color = color,width = 0.7)
plt.xlabel('X Label', fontsize=12)
plt.ylabel('Y Label', fontsize=12)
plt.xlabel("Meal Type")
plt.ylabel("Total")
plt.title("Grafik pemesanan setiap paket makanan terlaris", fontsize=16, fontweight='bold', y=1.01)
plt.show()
st.plotly(fig)

# # Marketing

marketing = dataset2[['Booking_ID', 'market_segment_type', 'room_type_reserved', 'arrival_year', 'arrival_month', 'arrival_date']]
marketing.head()

marketing['market_segment_type'].value_counts()

marketing.hist()

marketing_year2017 = marketing[marketing['arrival_year']==2017]
marketing_year2017.info()

marketing_year2017_groupBy = marketing_year2017[['market_segment_type', 'Booking_ID']].groupby(by='market_segment_type').count()
marketing_year2017_groupBy

x = list(marketing_year2017_groupBy.index)
y = marketing_year2017_groupBy['Booking_ID']
fig = plt.figure(figsize = (19, 10))
plt.bar(x, y, color = color,width = 0.7)
plt.xlabel('X Label', fontsize=12)
plt.ylabel('Y Label', fontsize=12)
plt.xlabel("Room Type")
plt.ylabel("Sum of Price (*10^6)")
plt.title("Grafik keuntungan setiap tipe kamar", fontsize=16, fontweight='bold', y=1.01)
plt.show()
st.plotly(fig)



marketing_year2018 = marketing[marketing['arrival_year']==2018]
marketing_year2018.info()

marketing_year2018_groupBy = marketing_year2018[['market_segment_type', 'Booking_ID']].groupby(by='market_segment_type').count()
marketing_year2018_groupBy

x = list(marketing_year2018_groupBy.index)
y = marketing_year2018_groupBy['Booking_ID']
fig = plt.figure(figsize = (19, 10))
plt.bar(x, y, color = color,width = 0.7)
plt.xlabel('X Label', fontsize=12)
plt.ylabel('Y Label', fontsize=12)
plt.xlabel("Room Type")
plt.ylabel("Sum of Price (*10^6)")
plt.title("Grafik keuntungan setiap tipe kamar", fontsize=16, fontweight='bold', y=1.01)
plt.show()
st.plotly(fig)

# ## revisi
# buat perbandingan antara (no_of_adults dengan no_of_children) serta perbandingan antara (no_of_week_nights dengan no_of_weekend_nights) 
#
# type_of_meal_plan dapat dibuat jadi pie chart atau bar chart,
#
# room type dapat dibuat jadi bar chart
#
# buat visualisasi market_segment_type

TARGET = 'room_type_reserved'


def select_columns(dataset: pd.DataFrame, threshold: int = 10) -> list[str]:
    """
    Function used to divide the columns of a DataFrame in discrete and continuous based on the specified threshold.

    Args:
        df (pd.DataFrame): A DataFrame containing the columns to be selected.
        threshold (int): A threshold defining the minimum number of unique values. Default is 10.

    Returns:
        discrete_columns (list[str]): A list of discrete column names.
        continuous_columns (list[str]): A list of continuous column names.
    """
    
    # Initialize the discrete columns list
    discrete_columns = []

    for column in dataset.columns:
        # Select the columns with number of unique values less than or equal to the threshold
        if dataset[column].nunique() <= threshold:
            # Append the selected columns to the list of discrete columns
            discrete_columns.append(column)
    
    # Select the continuous columns
    continuous_columns = [col for col in dataset.columns.tolist() if col not in discrete_columns]


    return discrete_columns, continuous_columns


def visualizations_discrete(dataset: pd.DataFrame):
    """
    Function used to plot countplots for each discrete column.

    Parameters:
        - df (pd.DataFrame): A DataFrame containing the data.
    """

    # Set the figure size and layout
    fig, axes = plt.subplots(nrows=len(discrete_columns), ncols=2, figsize=(12, 6 * len(discrete_columns)))
    fig.subplots_adjust(hspace=0.25)
    sns.color_palette("Blues", as_cmap=True)
    
    for index, column in enumerate(discrete_columns):
        # Plot countplot
        sns.countplot(data=dataset, x=column, ax=axes[index, 0], palette=color)
        axes[index, 0].set_title(f"Countplot of {column}")
        axes[index, 0].set_xlabel("")

        # Plot countplot
        sns.countplot(data=dataset, x=column, hue=TARGET, ax=axes[index, 1], palette=color)
        axes[index, 1].set_title(f"Countplot of {column} specified by {TARGET}")
        axes[index, 1].set_xlabel("")  

    # Show the plot
    plt.show()
    st.plotly(fig)


discrete_columns, continuous_columns = select_columns(dataset1)

visualizations_discrete(dataset1)

# +
colunas = dataset1['arrival_month'].value_counts()

plt.figure(figsize=(12,40))
for i, col in enumerate(colunas.index):
    ax = plt.subplot(12,2,i+1)
    sns.barplot(x='market_segment_type',
                y='avg_price_per_room', palette=color,
                hue='booking_status', data=dataset1[dataset1['arrival_month'] == col])
    plt.title(f'Bulan {col}',fontweight='bold')
plt.tight_layout()
plt.show()
st.plotly(colunas)

# +
colunas = ['no_of_adults','no_of_children','no_of_weekend_nights',
'no_of_week_nights','type_of_meal_plan','required_car_parking_space',
'room_type_reserved','arrival_year','arrival_month',
'market_segment_type','repeated_guest','no_of_previous_cancellations',
'no_of_special_requests','booking_status']

plt.figure(figsize=(10,40))
for i,col in enumerate(colunas):
    ax = plt.subplot(14,1,i+1)
    sns.countplot(y=dataset1[col],palette=color)
    plt.title(col)
    plt.ylabel(None)
plt.tight_layout()
plt.show()
st.plotly(colunas)

# # +
# colunas = ['no_of_adults','no_of_children','total']

# plt.figure(figsize=(14,10))
# for i,col in enumerate(colunas):
#     ax=plt.subplot(3,3,i+1)
#     sns.barplot(x=df[col],y=df['no_of_weekend_nights'], errorbar=None,palette=color)
#     plt.title(col, fontweight='bold')
# plt.tight_layout()
# plt.show()
