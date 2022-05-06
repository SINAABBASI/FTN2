import matplotlib.pyplot as plt



rng = range(4,13,2)

uncoded6_SD = [0.0404, 0.017024793388429754, 0.002578268876611418, 0.00026040995230114736, 5.66213743e-06]
uncoded5_SD = [0.07172414, 0.03961905, 0.00739927, 0.00101771, 5.36256457e-05]



# 13 obs
obs13_6=[0.051219512195121955,0.016880081300813005,0.004278455284552845,0.0004085365853658537,5.397148676171079e-05]
obs13_5=[0.09470468431771893,0.04675152749490835,0.013951120162932791,0.0026680244399185336,0.00026272912423625254]

# 15 obs
obs15_6=[0.057433808553971476,0.018228105906313647,0.0032790224032586564,0.0003971486761710794,2.0875763747454175e-05]
obs15_5=[0.09010204081632653,0.04408163265306122,0.012030612244897957,0.0019183673469387757,0.00017448979591836735]




plt.yscale("log")
plt.xlabel("Eb/N0")
plt.ylabel("BER")
plt.plot(rng,uncoded6_SD,'rx-')
plt.plot(rng,uncoded5_SD,'bx-')
plt.plot(rng,obs13_6,'ro--')
plt.plot(rng,obs13_5,'bo--')
plt.plot(rng,obs15_6,'rs--')
plt.plot(rng,obs15_5,'bs--')
plt.plot([12],1.3775510204081633e-05,'go')
plt.legend(['0.6 SD, ISI length 5','0.5 SD, ISI length 7',
            '13obs 0.6','13obs 0.5',
            '15obs 0.6','15obs 0.5','17obs'])
plt.show()

# number of cluster points: 131072
# cluster points shape (131072, 13)
# 6.758130081300814e-05

# number of cluster points: 524288
# cluster points shape (524288, 15)
# 2.0875763747454175e-05

# number of cluster points: 2097152
# cluster points shape (2097152, 17)
# 1.3775510204081633e-05

plt.title('12db, tau=0.6, ISI length=5')
plt.yscale("log")
plt.xlabel("Observation at reciver")
plt.ylabel("BER")
plt.plot(range(3,19,2), [0.02438128772635815,0.004087701612903226,0.0010858585858585857, 0.0003795546558704453,
                         0.00015618661257606493, 5.397148676171079e-05, 2.0875763747454175e-05,1.3775510204081633e-05],'bo-')

plt.show()


plt.title('12db,tau=0.6 - 13 observation at reciever')
plt.yscale("log")
plt.xlabel("ISI channel length in sys model")
plt.ylabel("BER")
plt.plot([3,5,7,9],[0.0010289046653144016, 5.397148676171079e-05, 4.5824847250509165e-05, 4.2857142857142856e-05],'ro-')
plt.show()