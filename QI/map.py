import csv

file = open('SP500.csv')
cont = csv.reader(file)



def MAP(pre,next):
    # average_day = 10
    price = []
    n_deal = 0.001
    had = 0
    init_money = 1000000
    money = init_money
    in_day = 0
    out_day = 0
    max_cash = 0
    least_day = min(pre,next)+1
    for idx, item in enumerate(cont):
        if idx == 0 or item[-1] == '':
            continue

        last_pr = float(item[6])
        price.append(last_pr)
        if idx <= least_day:
            continue

        if sum(price[-pre:]) / pre > sum(price[-next:]) / next:
            had += money / last_pr * n_deal
            money *= (1. - n_deal)
            in_day += 1
        else:
            money += last_pr * had * n_deal
            had *= (1. - n_deal)
            out_day += 1

    print("Dealing with {} days in and {} days out".format(in_day, out_day))
    print("Delta of cash : {}".format(money - init_money))
    print("Number of stock : {}".format(had))
    total = money + had * price[-1]
    print("Delta of funding : {}".format(total / init_money - 1.))

MAP(40,100)
# earing = 0
# pre_price = 0
# first_price = 0
# for idx, item in enumerate(cont):
#     if idx > 0 and item[-1] == '':
#         continue
#     if idx == 1:
#         print(item)
#         pre_price = float(item[6])
#         first_price = float(item[6])
#     if idx>1:
#         price = float(item[6])
#         if pre_price<price:
#             earing+=(price-pre_price)
#         pre_price = price
# print(earing,first_price)
# print(earing/first_price)