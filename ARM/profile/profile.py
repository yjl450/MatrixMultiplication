f = open("profile/profile.txt")
k84 = 0
k44 = 0
naive = 0
cnt = {}
for l in f.readlines():
    l = l.strip()
    if "*" in l:    
        cnt[l] = cnt.get(l, 0) + 1
    if "#" in l:
        if "#8x4" in l:
            k84 += 1
        elif "#4x4" in l:
            k44 += 1
        else:
            naive += 1

print(cnt)

total = k84 + k44 + naive
print("8*4:", k84/total)
print("4*4:", k44/total)
print("naive:", naive/total)