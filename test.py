class RandSSP:
    def __init__(self, seed=123456789):
        self.m = 2**31
        self.a = 2**16 + 3
        self.c = 0
        self.x = seed

    def generate(self, p, q):
        r = np.zeros((p, q))
        for l in range(q):
            for k in range(p):
                self.x = (self.a * self.x + self.c) % self.m
                r[k, l] = self.x / self.m
        return r

# Create generator instance
generator = RandSSP()

# Generate 3D data
r = generator.generate(3, 1500)
mydatax, mydatay, mydataz = r[0, :], r[1, :], r[2, :]

# Create 3D scatter plot
fig4 = px.scatter_3d(
    x=mydatax, y=mydatay, z=mydataz, color=mydataz,
    labels={'x': 'X Axis', 'y': 'Y Axis', 'z': 'Z Axis'}
)
fig4.update_traces(marker=dict(size=2))  # Smaller points
fig4.update_layout(
    title="3D Scatter Plot",
    width=plot_width,
    height=plot_height
)

# Export to HTML and open
pyo.plot(fig4, filename='3dscatter.html', auto_open=True)
