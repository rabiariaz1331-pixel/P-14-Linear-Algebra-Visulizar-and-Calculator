# Linear Algebra Visualizer & Calculator
# Single-file Streamlit app
# Save as: linear_algebra_visualizer.py
# Run: pip install -r requirements.txt
#      streamlit run linear_algebra_visualizer.py
import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Must be first Streamlit call
st.set_page_config(page_title="Linear Algebra Visualizer & Calculator", layout="wide")

def parse_matrix(text):
    """Parse a matrix from multiline text where rows are newline-separated and entries by spaces or commas."""
    try:
        rows = [r.strip() for r in text.strip().splitlines() if r.strip()]
        mat = [ [float(x) for x in r.replace(',', ' ').split()] for r in rows ]
        return np.array(mat)
    except Exception as e:
        raise ValueError("Could not parse matrix. Use rows on new lines and spaces or commas to separate entries.")


def matrix_to_sympy(A):
    return sp.Matrix(A.tolist())


def solve_linear_system(A, b):
    # use sympy for exact steps
    A_sym = sp.Matrix(A.tolist())
    b_sym = sp.Matrix(b.tolist())
    try:
        sol = A_sym.LUsolve(b_sym)
        return sol
    except Exception:
        # fallback to linsolve
        solset = sp.linsolve((A_sym, b_sym))
        return solset


def rref_steps(A, b=None):
    """Return step-wise row reduction using sympy's Matrix.rref info (we'll show the rref and rank)."""
    M = matrix_to_sympy(A)
    if b is not None:
        M = M.row_join(matrix_to_sympy(b))
    rref_matrix, pivotcols = M.rref()
    return rref_matrix, pivotcols


def grid_points(n=11, span=2.0):
    xs = np.linspace(-span, span, n)
    ys = np.linspace(-span, span, n)
    pts = np.array([[x,y] for x in xs for y in ys])
    return pts


def transform_grid_2x2(A, pts):
    return (A @ pts.T).T


def plot_2d_transformation(A, vectors=None):
    # A: 2x2
    pts = grid_points(n=11, span=3.0)
    pts_trans = transform_grid_2x2(A, pts)

    fig = go.Figure()
    # original grid points as small markers
    fig.add_trace(go.Scatter(x=pts[:,0], y=pts[:,1], mode='markers', marker=dict(size=3), name='Original grid'))
    fig.add_trace(go.Scatter(x=pts_trans[:,0], y=pts_trans[:,1], mode='markers', marker=dict(size=3), name='Transformed grid'))

    # axes
    fig.update_xaxes(range=[-5,5], zeroline=True)
    fig.update_yaxes(range=[-5,5], zeroline=True)

    # vectors
    if vectors:
        for i, v in enumerate(vectors):
            v = np.array(v, dtype=float).reshape(2,)
            fig.add_trace(go.Scatter(x=[0, v[0]], y=[0, v[1]], mode='lines+markers', name=f'v{i+1}', marker=dict(size=6)))
            v_t = A @ v
            fig.add_trace(go.Scatter(x=[0, v_t[0]], y=[0, v_t[1]], mode='lines+markers', name=f'A v{i+1}', marker=dict(symbol='x', size=6)))

    fig.update_layout(title='2D Linear Transformation (grid and vectors)',width=700, height=600)
    return fig


def plot_3d_transformation(A, vectors=None):
    # A: 3x3
    # create a small cube of points
    rng = np.linspace(-1.5,1.5,7)
    pts = np.array([[x,y,z] for x in rng for y in rng for z in rng])
    pts_t = (A @ pts.T).T

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='markers', marker=dict(size=2), name='Original'))
    fig.add_trace(go.Scatter3d(x=pts_t[:,0], y=pts_t[:,1], z=pts_t[:,2], mode='markers', marker=dict(size=2), name='Transformed'))

    if vectors:
        for i, v in enumerate(vectors):
            v = np.array(v, dtype=float).reshape(3,)
            vt = A @ v
            fig.add_trace(go.Scatter3d(x=[0,v[0]], y=[0,v[1]], z=[0,v[2]], mode='lines+markers', name=f'v{i+1}'))
            fig.add_trace(go.Scatter3d(x=[0,vt[0]], y=[0,vt[1]], z=[0,vt[2]], mode='lines+markers', name=f'A v{i+1}'))

    fig.update_layout(scene=dict(aspectmode='data'), width=800, height=700, title='3D Linear Transformation')
    return fig

# ------------------------- UI -------------------------

st.title("Linear Algebra Visualizer & Calculator")
st.write("Interactive solver and visualizer for matrices, linear systems, vectors, eigen stuff, and transformations.")

# Sidebar controls
with st.sidebar:
    st.header("Input / Controls")
    mode = st.selectbox("Mode", ["Matrix Operations", "Solve Linear System", "Eigen / SVD", "Vector Ops / Transformations"])
    st.markdown("---")
    st.caption("You can paste matrices or type small ones. Use commas or spaces between entries and newline for rows.")

# Shared matrix input area
if mode in ("Matrix Operations", "Eigen / SVD", "Vector Ops / Transformations"):
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("Primary matrix (A)")
        default = "1 0\n0 1"
        A_text = st.text_area("Matrix A", value=default, height=120)
    with col2:
        st.subheader("Optional secondary matrix or vector (B)")
        B_text = st.text_area("Matrix B / vector (optional)", value="", height=120)

    # parse matrices
    try:
        A = parse_matrix(A_text)
    except Exception as e:
        st.error(str(e))
        st.stop()

    B = None
    if B_text.strip():
        try:
            B = parse_matrix(B_text)
        except Exception as e:
            st.error(str(e))
            st.stop()

    st.write("**A =**")
    st.dataframe(A)
    if B is not None:
        st.write("**B =**")
        st.dataframe(B)

    # Matrix features
    if mode == "Matrix Operations":
        st.subheader("Matrix Operations & Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"Shape: {A.shape}")
            st.write(f"Rank: {np.linalg.matrix_rank(A)}")
            try:
                det = float(np.linalg.det(A)) if A.shape[0]==A.shape[1] else None
            except Exception:
                det = None
            st.write(f"Determinant: {det}")
        with col2:
            if A.shape[0]==A.shape[1]:
                try:
                    inv = np.linalg.inv(A)
                    st.write("Inverse:")
                    st.dataframe(inv)
                except Exception as e:
                    st.write("Inverse: Not invertible or numeric issue")
            else:
                st.write("Inverse: Not square")
        with col3:
            st.write("Sympy exact form:")
            st.code(str(matrix_to_sympy(A)), language='python')

        # Visualize transformation if 2x2 or 3x3
        if A.shape == (2,2):
            st.plotly_chart(plot_2d_transformation(A), use_container_width=True)
        elif A.shape == (3,3):
            st.plotly_chart(plot_3d_transformation(A), use_container_width=True)

        # combine with B if present
        if B is not None:
            try:
                if B.shape == A.shape:
                    st.write("A + B:")
                    st.dataframe(A+B)
                    st.write("A * B (dot):")
                    st.dataframe(A.dot(B))
                elif B.size == A.shape[1]:
                    st.write("Treating B as vector -> A * B:")
                    st.dataframe(A.dot(B.reshape(-1,1)))
            except Exception as e:
                st.write("Could not combine A and B:", e)

    elif mode == "Eigen / SVD":
        st.subheader("Eigenvalues, Eigenvectors, and SVD")
        if A.shape[0] != A.shape[1]:
            st.warning("Eigen decomposition requires a square matrix.")
        else:
            try:
                eigvals, eigvecs = np.linalg.eig(A)
                st.write("Eigenvalues:")
                st.write(eigvals)
                st.write("Eigenvectors (columns):")
                st.dataframe(eigvecs)

                # Sympy exact eigen
                spA = matrix_to_sympy(A)
                st.write("Sympy eigen decomposition (exact / symbolic):")
                st.code(str(spA.eigenvects()), language='python')

                if A.shape == (2,2):
                    # show eigenvectors on plot
                    vectors = [eigvecs[:,i] for i in range(eigvecs.shape[1])]
                    st.plotly_chart(plot_2d_transformation(A, vectors=vectors), use_container_width=True)
                elif A.shape == (3,3):
                    vectors = [eigvecs[:,i] for i in range(eigvecs.shape[1])]
                    st.plotly_chart(plot_3d_transformation(A, vectors=vectors), use_container_width=True)
            except Exception as e:
                st.error(f"Could not compute eigen decomposition: {e}")

    elif mode == "Vector Ops / Transformations":
        st.subheader("Apply transformation to vectors")
        st.write("If B is a vector (n x 1) or list of vectors, we'll transform them using A.")
        if B is not None:
            if A.shape[1] != B.shape[1] and B.shape[0] == A.shape[1]:
                Bv = B.reshape(-1,1)
            else:
                Bv = B
            try:
                transformed = (A @ Bv.T).T if Bv.ndim==2 and Bv.shape[1]==1 else (A @ Bv.T).T
                st.write("Transformed vectors:")
                st.dataframe(transformed)
                if A.shape==(2,2):
                    vecs = [Bv[:,i] for i in range(Bv.shape[1])] if Bv.ndim==2 else [Bv]
                    st.plotly_chart(plot_2d_transformation(A, vectors=vecs), use_container_width=True)
                elif A.shape==(3,3):
                    vecs = [Bv[:,i] for i in range(Bv.shape[1])] if Bv.ndim==2 else [Bv]
                    st.plotly_chart(plot_3d_transformation(A, vectors=vecs), use_container_width=True)
            except Exception as e:
                st.error(e)

# Solve linear system mode
elif mode == "Solve Linear System":
    st.subheader("Solve linear systems of equations (Ax = b)")
    eq_text = st.text_area("Enter system or paste matrix A (rows) and vector b on right separated by |\nExample:\n2 1 | 5\n3 -1 | 4", height=200)
    if st.button("Solve"):
        try:
            lines = [line for line in eq_text.splitlines() if line.strip()]
            A_rows = []
            b_rows = []
            for line in lines:
                if '|' in line:
                    left, right = line.split('|')
                elif ',' in line and ';' in line:
                    left, right = line.split(';')
                else:
                    # assume last entry is b
                    parts = line.replace(',', ' ').split()
                    left = ' '.join(parts[:-1])
                    right = parts[-1]
                A_rows.append([float(x) for x in left.replace(',', ' ').split()])
                b_rows.append(float(right))
            A = np.array(A_rows)
            b = np.array(b_rows)
            st.write("A:")
            st.dataframe(A)
            st.write("b:")
            st.dataframe(b.reshape(-1,1))

            # Show rref and pivots
            rref_matrix, pivots = rref_steps(A, b)
            st.write("RREF (A | b):")
            st.text(str(rref_matrix))
            st.write(f"Pivot columns: {pivots}")

            # Solve
            try:
                sol = solve_linear_system(A, b)
                st.write("Solution:")
                st.write(sol)
            except Exception as e:
                st.error(f"Could not solve: {e}")

            # If 2x2 or 3x3 visualize
            if A.shape==(2,2):
                # treat equations as lines in 2D
                # use sympy to parse the equations for plotting
                # quick approximate: derive lines from A x = b
                import numpy as _np
                x = _np.linspace(-5,5,200)
                y1 = (b[0] - A[0,0]*x)/A[0,1] if A[0,1]!=0 else None
                y2 = (b[1] - A[1,0]*x)/A[1,1] if A[1,1]!=0 else None
                fig = go.Figure()
                if y1 is not None:
                    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Eq1'))
                if y2 is not None:
                    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Eq2'))
                if isinstance(sol, sp.Matrix) or hasattr(sol, 'evalf'):
                    sol_np = np.array([float(sol[i]) for i in range(len(sol))])
                    fig.add_trace(go.Scatter(x=[sol_np[0]], y=[sol_np[1]], mode='markers', name='Solution', marker=dict(size=8)))
                fig.update_layout(title='Lines representing equations', width=700, height=500)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error parsing system: {e}")



