import vectorbt as vbt

print("Keys in vbt.settings:")
for k in vbt.settings.keys():
    print(f"{k}: {vbt.settings[k]}")
