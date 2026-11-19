from django import forms


class BukuForm(forms.Form):
    user_rating = forms.FloatField(
        label="User Rating (1.0 - 5.0)",
        min_value=1.0, max_value=5.0, initial=4.7,
        widget=forms.NumberInput(attrs={"step": "0.1"}),
    )
    reviews = forms.IntegerField(
        label="Jumlah Reviews", min_value=0, max_value=100000, initial=15200,
    )
    price = forms.FloatField(
        label="Harga (USD)", min_value=0, max_value=150, initial=12,
    )
    year = forms.IntegerField(
        label="Tahun masuk bestseller (2009-2019)",
        min_value=2009, max_value=2019, initial=2018,
    )