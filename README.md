
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Sample Data
df_reference = pd.DataFrame({
    'rate1': np.random.randn(100),
    'rate2': np.random.randn(100),
    'rate3': np.random.randn(100),
})
df_product = pd.DataFrame({
    'product1': np.random.randn(100),
    'product2': np.random.randn(100)
})

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("PCA Regression Projector"),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select Reference Elements:"),
            dcc.Checklist(
                id='reference-elements',
                options=[{'label': i, 'value': i} for i in df_reference.columns],
                value=df_reference.columns.tolist(),
                inline=True
            )
        ]),
        dbc.Col([
            html.Label("Select Product to Regress:"),
            dcc.Dropdown(
                id='product-dropdown',
                options=[{'label': i, 'value': i} for i in df_product.columns],
                value=df_product.columns[0]
            )
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Projection on Eigenvectors:"),
            html.Div(id='projection-output')
        ])
    ]),
])

@app.callback(
    Output('projection-output', 'children'),
    Input('reference-elements', 'value'),
    Input('product-dropdown', 'value')
)
def update_projection(selected_elements, product_to_regress):
    # Filter based on selected elements
    filtered_df = df_reference[selected_elements]

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(filtered_df)

    # Compute PCA
    pca = PCA()
    pca.fit(scaled_data)

    # Project the product time series onto PCA eigenvectors
    product_data = df_product[product_to_regress].values.reshape(-1, 1)
    product_on_eigenvectors = np.dot(product_data - product_data.mean(), pca.components_.T)

    # Create a DataFrame for the projection results
    df_projection = pd.DataFrame(
        product_on_eigenvectors, 
        columns=[f'PC{i+1}' for i in range(pca.n_components_)]
    )
    
    # Return as an HTML table
    return dbc.Table.from_dataframe(df_projection, striped=True, bordered=True, hover=True)

if __name__ == '__main__':
    app.run_server(debug=True)
Run this code, and it will create a Dash web app. You can access it in your web browser by visiting http://127.0.0.1:8050/.
