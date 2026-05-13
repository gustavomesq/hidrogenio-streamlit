import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import simpson

# ======================================================
# Configuração da página
# ======================================================

st.set_page_config(
    page_title="Átomos Hidrogenoides",
    layout="wide"
)

st.title("Átomos Hidrogenoides — Equação de Schrödinger 3D")

st.markdown("""
Este aplicativo mostra distribuições radiais de probabilidade
para orbitais hidrogenoides.
""")

st.latex(r"P_{nl}(r)=r^2|R_{nl}(r)|^2")

# ======================================================
# Seção teórica
# ======================================================

with st.expander("Fundamentação Teórica", expanded=True):

    st.header("Equação de Schrödinger 3D e Átomos Hidrogenoides")

    st.markdown("""
A solução da equação de Schrödinger para átomos hidrogenoides
pode ser separada em uma parte radial e uma parte angular.
""")

    st.latex(r"\psi(r,\theta,\phi)=R_{nl}(r)Y_l^m(\theta,\phi)")

    st.markdown("""
onde:

- \(R_{nl}(r)\) representa a parte radial;
- \(Y_l^m(\theta,\phi)\) são os harmônicos esféricos.
""")

    # ==================================================
    # Elemento de volume
    # ==================================================

    st.subheader("Elemento de volume em coordenadas esféricas")

    st.latex(r"d\tau=r^2\sin\theta\,dr\,d\theta\,d\phi")

    st.markdown("""
O fator \(r^2\) surge naturalmente do elemento de volume esférico.

Por isso a distribuição radial de probabilidade é:
""")

    st.latex(r"P_{nl}(r)=r^2|R_{nl}(r)|^2")

    st.markdown("""
A quantidade \(P(r)dr\) representa a probabilidade
de encontrar o elétron entre \(r\) e \(r+dr\).
""")

    # ==================================================
    # Valor esperado
    # ==================================================

    st.subheader("Valor esperado do raio")

    st.markdown("""
O valor esperado da distância radial é dado pela integral tripla:
""")

    st.latex(
        r"\langle r\rangle=\iiint r|\psi(r,\theta,\phi)|^2d\tau"
    )

    st.markdown("""
Após separar a parte angular:
""")

    st.latex(
        r"\langle r\rangle=\int_0^\infty rP_{nl}(r)dr"
    )

    st.markdown("""
Para átomos hidrogenoides:
""")

    st.latex(
        r"\langle r \rangle_{nl}=\frac{a_0}{2}[3n^2-l(l+1)]"
    )

    # ==================================================
    # Momento angular
    # ==================================================

    st.subheader("Momento angular orbital")

    st.markdown("""
Os harmônicos esféricos são autofunções
dos operadores de momento angular orbital.
""")

    st.latex(
        r"L^2Y_l^m=\hbar^2l(l+1)Y_l^m"
    )

    st.latex(
        r"L_zY_l^m=m\hbar Y_l^m"
    )

    st.markdown("""
onde:

- \(l\) é o número quântico orbital;
- \(m\) é o número quântico magnético.
""")

    # ==================================================
    # Nós radiais
    # ==================================================

    st.subheader("Nós radiais")

    st.markdown("""
O número de nós radiais é:
""")

    st.latex(
        r"N_{\mathrm{nós}}=n-l-1"
    )

    # ==================================================
    # Raio mais provável
    # ==================================================

    st.subheader("Raio mais provável")

    st.markdown("""
Os máximos da distribuição radial correspondem
aos raios mais prováveis para encontrar o elétron.

Eles são obtidos impondo:
""")

    st.latex(
        r"\frac{dP_{nl}}{dr}=0"
    )

# ======================================================
# Funções das distribuições radiais
# ======================================================

def distribuicao_radial(n, l, r):

    # 1s
    if n == 1 and l == 0:
        P = 4 * r**2 * np.exp(-2*r)
        nome = "1s"

    # 2s
    elif n == 2 and l == 0:
        P = (1/8) * r**2 * (2 - r)**2 * np.exp(-r)
        nome = "2s"

    # 2p
    elif n == 2 and l == 1:
        P = (1/24) * r**4 * np.exp(-r)
        nome = "2p"

    # 3s
    elif n == 3 and l == 0:
        P = (4/19683) * r**2 * (27 - 18*r + 2*r**2)**2 * np.exp(-2*r/3)
        nome = "3s"

    # 3p
    elif n == 3 and l == 1:
        P = (8/19683) * r**4 * (6 - r)**2 * np.exp(-2*r/3)
        nome = "3p"

    # 3d
    elif n == 3 and l == 2:
        P = (4/32805) * r**6 * np.exp(-2*r/3)
        nome = "3d"

    else:
        return None, None

    return P, nome

# ======================================================
# Sidebar
# ======================================================

st.sidebar.header("Configurações")

modo = st.sidebar.selectbox(
    "Escolha o modo:",
    ["Orbital único", "Comparar orbitais"]
)

# eixo radial
r = np.linspace(0, 30, 4000)

# ======================================================
# MODO 1 — ORBITAL ÚNICO
# ======================================================

if modo == "Orbital único":

    estados = {
        "1s (1,0)": (1,0),
        "2s (2,0)": (2,0),
        "2p (2,1)": (2,1),
        "3s (3,0)": (3,0),
        "3p (3,1)": (3,1),
        "3d (3,2)": (3,2)
    }

    escolha = st.sidebar.selectbox(
        "Escolha o orbital:",
        list(estados.keys())
    )

    n, l = estados[escolha]

    P, nome = distribuicao_radial(n, l, r)

    # ==================================================
    # Máximos
    # ==================================================

    indices_maximos, _ = find_peaks(P)

    r_maximos = r[indices_maximos]
    P_maximos = P[indices_maximos]

    # ==================================================
    # Nós radiais
    # ==================================================

    nos_teoricos = n - l - 1

    # ==================================================
    # Valor esperado <r>
    # ==================================================

    r_medio_numerico = simpson(r * P, r)

    r_medio_teorico = 0.5 * (3*n**2 - l*(l+1))

    # ==================================================
    # Informações quânticas
    # ==================================================

    autovalor_L2 = l*(l+1)

    valores_m = list(range(-l, l+1))

    # ==================================================
    # Informações gerais
    # ==================================================

    st.subheader(f"Orbital {nome}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Número quântico n", n)
        st.metric("Número quântico l", l)

    with col2:
        st.metric("Nós radiais", nos_teoricos)
        st.metric("Número de máximos", len(r_maximos))

    with col3:
        st.metric("<r> numérico", f"{r_medio_numerico:.3f} a₀")
        st.metric("<r> teórico", f"{r_medio_teorico:.3f} a₀")

    # ==================================================
    # Momento angular orbital
    # ==================================================

    st.subheader("Momento angular orbital")

    st.latex(r"L^2Y_l^m = \hbar^2 l(l+1)Y_l^m")
    st.latex(r"L_zY_l^m = m\hbar Y_l^m")

    st.markdown(f"""
- Para este orbital, o número quântico orbital é:

    l = {l}

- O autovalor de \(L^2\) é:

    \(\hbar^2 l(l+1) = {autovalor_L2}\hbar^2\)

- Os valores possíveis do número quântico magnético são:

    m = {valores_m}

- Portanto, os valores possíveis de \(L_z\) são:
""")

    for m in valores_m:
        st.write(f"Lz = {m}ℏ")

    # ==================================================
    # Máximos da distribuição
    # ==================================================

    st.subheader("Máximos da distribuição radial")

    for i in range(len(r_maximos)):
        st.write(f"Máximo {i+1}: r = {r_maximos[i]:.3f} a₀")

    # ==================================================
    # Gráfico
    # ==================================================

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(r, P, linewidth=2, label=nome)

    ax.scatter(r_maximos, P_maximos, s=60)

    for i in range(len(r_maximos)):
        ax.text(
            r_maximos[i],
            P_maximos[i],
            f"{r_maximos[i]:.2f}"
        )

    ax.set_xlabel("r/a₀")
    ax.set_ylabel("P(r)")
    ax.set_title(f"Distribuição radial — {nome}")

    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

    # ==================================================
    # Interpretação física
    # ==================================================

    st.subheader("Interpretação física")

    st.markdown(f"""
- O gráfico representa a distribuição radial de probabilidade do orbital **{nome}**.
- Os máximos indicam os raios mais prováveis para encontrar o elétron.
- O número de nós radiais é dado por:

    n - l - 1 = {nos_teoricos}

- O valor esperado ⟨r⟩ representa a distância média do elétron ao núcleo.
""")

# ======================================================
# MODO 2 — COMPARAÇÃO
# ======================================================

elif modo == "Comparar orbitais":

    st.subheader("Comparação entre orbitais")

    opcoes = {
        "1s": (1,0),
        "2s": (2,0),
        "2p": (2,1),
        "3s": (3,0),
        "3p": (3,1),
        "3d": (3,2)
    }

    selecionados = st.multiselect(
        "Escolha os orbitais:",
        list(opcoes.keys()),
        default=["1s", "2s", "2p"]
    )

    fig, ax = plt.subplots(figsize=(10,5))

    for item in selecionados:

        n, l = opcoes[item]

        P, nome = distribuicao_radial(n, l, r)

        ax.plot(r, P, linewidth=2, label=nome)

    ax.set_xlabel("r/a₀")
    ax.set_ylabel("P(r)")
    ax.set_title("Comparação entre distribuições radiais")

    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

    st.markdown("""
### Observações físicas

- Conforme n aumenta, o orbital se torna mais espalhado.
- Orbitais s possuem maior penetração perto do núcleo.
- Orbitais p e d possuem máximos mais afastados.
- O número de nós aumenta com n.
""")

# ======================================================
# Rodapé
# ======================================================

st.markdown("---")

st.markdown("""
Projeto desenvolvido para estudo da Equação de Schrödinger 3D
e átomos hidrogenoides.
""")