import csv

file = open('SP500.csv')
cont = csv.reader(file)

average_day = 40
price = []
n_deal = 0.001
had = 0
init_money = 1000000
money = init_money
in_day = 0
out_day = 0
max_cash = 0

for idx, item in enumerate(cont):
    if idx == 0 or item[-1] == '':
        continue

    last_pr = float(item[6])
    price.append(last_pr)
    if idx <= average_day:
        continue

    if last_pr < sum(price[-average_day:]) / average_day:
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