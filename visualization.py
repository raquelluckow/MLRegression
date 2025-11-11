import plotly.express as px
import pandas as pd
from model import y_pred, y_test

df = pd.DataFrame({
    'True Values': y_test,
    'Predicted Values': y_pred
})

fig = px.scatter(df, x='True Values', y='Predicted Values', 
                 title="True vs Predicted Diabetes Progression", 
                 labels={'True Values': 'True Values', 'Predicted Values': 'Predicted Values'})

fig.update_traces(marker=dict(size=8, color='rgba(255, 99, 132, 0.6)', symbol='circle', line=dict(width=2, color='black')))

fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
              line=dict(color="black", dash="dot", width=3))
fig.update_layout(
    title="True vs Predicted Diabetes Progression", 
    title_font=dict(size=20, color="darkblue"),  
    xaxis=dict(title="True Values", title_font=dict(size=14, color="black"), showgrid=True, zeroline=True),  
    yaxis=dict(title="Predicted Values", title_font=dict(size=14, color="black"), showgrid=True, zeroline=True),  
    plot_bgcolor='whitesmoke',  # Background color
    font=dict(family="Arial", size=12, color="black"),  
    margin=dict(l=40, r=40, t=40, b=40),  
    showlegend=False 
)

print(fig.show())
fig.show()