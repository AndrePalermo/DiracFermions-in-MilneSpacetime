import mpmath as mp
import numpy as np
import scipy as spy

mp.dps = 20

#####################       SPECIAL FUNCTIONS FOR CALCULATIONS      ########################

def FAST_specialfunctions(τ, μ, mass, px, py):
    # returns fhw, jw, sw, tw
    exp_ = mp.exp(-mp.pi * μ)
    h1_m12p = mp.hankel1(-1/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
    h1_p12p = mp.hankel1(1/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
    h1_p12m = mp.hankel1(1/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
    h1_m12m = mp.hankel1(-1/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
    h2_m32m = mp.hankel2(-3/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
    h2_p12m = mp.hankel2(1/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
    h2_m12m = mp.hankel2(-1/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
    h2_p32m = mp.hankel2(3/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
    h2_m32p = mp.hankel2(-3/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
    h2_p12p = mp.hankel2(1/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
    h2_m12p = mp.hankel2(-1/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
    h2_p32p = mp.hankel2(3/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
    
    hw = (1/8) * exp_ * mp.pi * (
        h1_m12p *
        (h2_m32m - h2_p12m) +
        h1_p12p *
        (h2_m12m - h2_p32m)
    )
    
    jw = -(1/8) * mp.pi * (
        h2_m12m *
        (h2_m32p - h2_p12p) +
        h2_p12m *
        (h2_m12p - h2_p32p)
    )
    
    sw = (mp.pi / 4) * exp_ * (
        h1_p12p * h2_p12m - h1_m12p * h2_m12m
    )
    
    tw = -(mp.pi / 2) * (
            h2_p12m * h2_p12p
        )
    
    fw = mp.pi/4 *exp_*(
        h1_m12p*h2_p12m
    )
    
    ww = mp.pi/4 * (
        h1_m12p*h1_p12m+h1_p12p*h1_m12m
    )
    
    zw = mp.pi/4 *(
        h1_p12p*h1_m12m-h1_m12p*h1_p12m
    )
    
    return -mp.im(hw), jw, sw, tw, fw, ww, zw

def fhw(τ, μ, mass, px, py):
    """
        Returns mathfrak{h}_w
    """
    #Apparently this function is stable only for mu>0. For mu<0 it blows up at mu~-15. The next line uses parity to prevent the problem
    μ = np.abs(μ)
    # if(mp.sqrt(mass**2 + px**2 + py**2) * τ > 10):
        # return 1/(mp.sqrt(mass**2 + px**2 + py**2) * τ+1e-10)
    
    result = (1/16) * 1j * mp.exp(-mp.pi * μ) * mp.pi * (
        -mp.hankel1(-3/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) * mp.hankel2(-1/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
        + 2 * mp.hankel1(1/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) * mp.hankel2(-1/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
        + mp.hankel1(-1/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) * (
            mp.hankel2(-3/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
            - 2 * mp.hankel2(1/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
        )
        + mp.hankel1(3/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) * mp.hankel2(1/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
        - mp.hankel1(1/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) * mp.hankel2(3/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
    )
    return result

def hw(τ, μ, mass, px, py):
    """
        Returns h_w
    """
    result = (1/8) * mp.exp(-mp.pi * μ) * mp.pi * (
        mp.hankel1(-1/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) *
        (mp.hankel2(-3/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) -
        mp.hankel2(1/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)) +
        mp.hankel1(1/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) *
        (mp.hankel2(-1/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) -
        mp.hankel2(3/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ))
    )
    return result

def chw(τ, μ, mass, px, py):
    """
        Returns h_w^* (i.e. the conjugate of h_w)
    """
    result = (1/8) * mp.exp(-mp.pi * μ) * mp.pi * (
        mp.hankel1(-3/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) * mp.hankel2(-1/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
        - mp.hankel1(1/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) * mp.hankel2(-1/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
        + (mp.hankel1(-1/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) - mp.hankel1(3/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)) * mp.hankel2(1/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
    )
    return result

def jw(τ, μ, mass, px, py):
    """
        Returns j_w
    """
    # if(mp.sqrt(mass**2 + px**2 + py**2) * τ > 10):
        # return mp.exp(-2*1j*mp.sqrt(mass**2 + px**2 + py**2) * τ)/(mp.sqrt(mass**2 + px**2 + py**2) * τ)**2
    
    result = -(1/8) * mp.pi * (
        mp.hankel2(-1/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) *
        (mp.hankel2(-3/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) -
        mp.hankel2(1/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)) +
        mp.hankel2(1/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) *
        (mp.hankel2(-1/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) -
        mp.hankel2(3/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ))
    )
    return result

def cjw(τ, μ, mass, px, py):
    """
        Returns j_w^* (i.e. the conjugate of j_ww)
    """
    # if(mp.sqrt(mass**2 + px**2 + py**2) * τ > 10):
        # return mp.exp(2*1j*mp.sqrt(mass**2 + px**2 + py**2) * τ)/(mp.sqrt(mass**2 + px**2 + py**2) * τ)**2
    
    result = -(1/8) * mp.pi * (
        mp.hankel1(-3/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) * mp.hankel1(-1/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
        - mp.hankel1(1/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) * mp.hankel1(-1/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
        + (mp.hankel1(-1/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) - mp.hankel1(3/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)) * mp.hankel1(1/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
    )
    return result

def sw(τ, μ, mass, px, py):
    """
        Returns s_w
    """
    # if(mp.sqrt(mass**2 + px**2 + py**2) * τ > 10):
        # return -μ/(mp.sqrt(mass**2 + px**2 + py**2) * τ)**2
    
    
    result = (mp.pi / 4) * mp.exp(-μ * mp.pi) * (
        mp.hankel1(1/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) * mp.hankel2(1/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
        - mp.hankel1(-1/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) * mp.hankel2(-1/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
    )
    return result

def tw(τ, μ, mass, px, py):
    """
        Returns t_w
    """
    # if(mp.sqrt(mass**2 + px**2 + py**2) * τ > 10):
        # return -mp.exp(-2*1j*mp.sqrt(mass**2 + px**2 + py**2) * τ)/(mp.sqrt(mass**2 + px**2 + py**2) * τ)
    
    result = -(mp.pi / 2) * (
            mp.hankel2(1/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) * mp.hankel2(1/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
        )
    return result

def ctw(τ, μ, mass, px, py):
    """
        Returns t_w^* (i.e. the conjugate of t_w)
    """
    # if(mp.sqrt(mass**2 + px**2 + py**2) * τ > 10):
        # return -mp.exp(2*1j*mp.sqrt(mass**2 + px**2 + py**2) * τ)/(mp.sqrt(mass**2 + px**2 + py**2) * τ)
    
    result = -(mp.pi / 2) * (
            mp.hankel1(1/2 + 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ) * mp.hankel1(1/2 - 1j*μ, mp.sqrt(mass**2 + px**2 + py**2) * τ)
        )
    return result


#####################       DIAGONALIZATION      ########################


def eigenvals(τ, μ, pm1, pm2, β, SP, mass, pT):
    """
        returns the eigenvalues corresponding to the thermodynamic and kinematic parameters. pm1 and pm2 must be either 1 or -1
    """
    mT=mp.sqrt(mass**2+pT**2)
    term1 = β**2 * (mT**2 + μ**2 / τ**2)
    term2 = SP**2 / 4
    term3 = pm2 * abs(SP) * β * mp.sqrt(mass**2 + μ**2 / τ**2)

    return pm1 * mp.sqrt(term1 + term2 + term3)

# def compute_normalized_vector_plusplus(τ, μ, mass, px, py, β, SP):
#     """
#         returns the eigenvector corresponding to +E_+. 
        
#         !!! WARNING: this only works if px^2+py^2 != 0 !!!
#             if px^2 + py^2 == 0 it is not an eigenvector
#     """
#     s=mp.sign(SP)
#     pT_sq = px**2 + py**2
#     mL = mp.sqrt(mass**2+μ**2/τ**2)

#     eig_val = eigenvals(τ, μ, 1, 1, β, SP, mass, mp.sqrt(pT_sq))

#     fhw_val = fhw(τ, μ, mass, px, py)
#     jw_val = jw(τ, μ, mass, px, py)
#     cjw_val = mp.conj(jw_val)
#     sw_val = sw(τ, μ, mass, px, py)
#     tw_val = tw(τ, μ, mass, px, py)

#     mT = mp.sqrt(mass**2 + pT_sq)

#     # First component
#     comp1 = (
#         β * (mass * (-mass**2 + mL**2 + mT**2) * s + mL * mT**3 * τ * fhw_val) 
#     ) / (
#         2 * mass * mT**2 * s * τ * eig_val * jw_val
#     ) + (
#         mL + mass * mT * s * τ * fhw_val
#     ) / (
#         2 * mass * mT * s * τ * jw_val
#     ) + (
#         SP * (mass * mL + mass**2 * mT * s * τ * fhw_val - pT_sq * s * μ * sw_val)
#     ) / (
#         4 * mass * mT**2 * s * τ * eig_val * jw_val
#     )

#     # Second component
#     comp2 = (
#         (px + 1j * py) * μ / (2 * mass * mT**2 * τ**2 * jw_val)
#         + (px + 1j * py) * β * μ * fhw_val / (2 * mass * τ * eig_val * jw_val)
#         + (px + 1j * py) * SP * (
#             mT**2 * τ**2 * (-mL + mass * mT * s * τ * fhw_val) * sw_val
#             + mass * s * (μ + 1j * mT**3 * τ**3 * cjw_val * tw_val)
#         ) / (4 * mass * mT**3 * s * τ**2 * eig_val * jw_val)
#     )

#     # Third component
#     comp3 = (
#         (px + 1j * py) * β * μ
#     ) / (
#         2 * mass * τ * eig_val
#     ) + (
#         (px + 1j * py) * SP * (mass * mT * s * τ * jw_val * sw_val - 1j * (mL + mass * mT * s * τ * fhw_val) * tw_val)
#     ) / (
#         4 * mass * mT * s * eig_val * jw_val
#     )

#     # Fourth component
#     comp4 = (
#         1/2 + (mL * mT * β) / (2 * mass * s * eig_val) + (
#             SP * (mass**2 * mT * τ * jw_val - 1j * pT_sq * μ * tw_val)
#         ) / (
#             4 * mass * mT**2 * τ * eig_val * jw_val
#         )
#     )

#     vec = [
#         mp.mpc(comp1),
#         mp.mpc(comp2),
#         mp.mpc(comp3),
#         mp.mpc(comp4)
#     ]

#     norm = mp.sqrt(sum(vi * mp.conj(vi) for vi in vec))
#     normalized_vec = [vi / norm for vi in vec]

#     return normalized_vec

# def compute_normalized_vector_plusminus(τ, μ, mass, px, py, β, SP):
#     """
#         returns the eigenvector corresponding to +E_-. 
        
#         !!! WARNING: this only works if px^2+py^2 != 0 !!!
#             if px^2 + py^2 == 0 it is not an eigenvector
#     """
#     s = mp.sign(SP)
#     pT_sq = px**2 + py**2
#     mL = mp.sqrt(mass**2 + μ**2 / τ**2)

#     eig_val = eigenvals(τ, μ, 1, -1, β, SP, mass, mp.sqrt(pT_sq))

#     fhw_val = fhw(τ, μ, mass, px, py)
#     jw_val = jw(τ, μ, mass, px, py)
#     cjw_val = mp.conj(jw_val)
#     sw_val = sw(τ, μ, mass, px, py)
#     tw_val = tw(τ, μ, mass, px, py)
    
#     mT = mp.sqrt(mass**2 + pT_sq)

#     # First component
#     comp1 = (
#         β * (mass * (-mass**2 + mL**2 + mT**2) * s - mL * mT**3 * τ * fhw_val)
#     ) / (
#         2 * mass * mT**2 * s * τ * eig_val * jw_val
#     ) - (
#         mL - mass * mT * s * τ * fhw_val
#     ) / (
#         2 * mass * mT * s * τ * jw_val
#     ) - (
#         SP * (mass * mL - mass**2 * mT * s * τ * fhw_val + pT_sq * s * μ * sw_val)
#     ) / (
#         4 * mass * mT**2 * s * τ * eig_val * jw_val
#     )

#     # Second component
#     comp2 = (
#         (px + 1j * py) * μ / (2 * mass * mT**2 * τ**2 * jw_val)
#         + (px + 1j * py) * β * μ * fhw_val / (2 * mass * τ * eig_val * jw_val)
#         + (px + 1j * py) * SP * (
#             mT**2 * τ**2 * (mL + mass * mT * s * τ * fhw_val) * sw_val
#             + mass * s * (μ + 1j * mT**3 * τ**3 * cjw_val * tw_val)
#         ) / (4 * mass * mT**3 * s * τ**2 * eig_val * jw_val)
#     )

#     # Third component
#     comp3 = (
#         (px + 1j * py) * β * μ
#     ) / (
#         2 * mass * τ * eig_val
#     ) + (
#         (px + 1j * py) * SP * (mass * mT * s * τ * jw_val * sw_val + 1j * (mL - mass * mT * s * τ * fhw_val) * tw_val)
#     ) / (
#         4 * mass * mT * s * eig_val * jw_val
#     )

#     # Fourth component
#     comp4 = (
#         1/2 - (mL * mT * β) / (2 * mass * s * eig_val) + (
#             SP * (mass**2 * mT * τ * jw_val - 1j * pT_sq * μ * tw_val)
#         ) / (
#             4 * mass * mT**2 * τ * eig_val * jw_val
#         )
#     )

#     vec = [
#         mp.mpc(comp1),
#         mp.mpc(comp2),
#         mp.mpc(comp3),
#         mp.mpc(comp4)
#     ]

#     norm = mp.sqrt(sum(vi * mp.conj(vi) for vi in vec))
#     normalized_vec = [vi / norm for vi in vec]

#     return normalized_vec

# def compute_normalized_vector_minusplus(τ, μ, mass, px, py, β, SP):
#     """
#         returns the eigenvector corresponding to -E_+. 
        
#         !!! WARNING: this only works if px^2+py^2 != 0 !!!
#             if px^2 + py^2 == 0 it is not an eigenvector
#     """
#     s = mp.sign(SP)
#     pT_sq = px**2 + py**2
#     mL = mp.sqrt(mass**2 + μ**2 / τ**2)

#     eig_val = eigenvals(τ, μ, 1, 1, β, SP, mass, mp.sqrt(pT_sq))

#     fhw_val = fhw(τ, μ, mass, px, py)
#     jw_val = jw(τ, μ, mass, px, py)
#     cjw_val = mp.conj(jw_val)
#     sw_val = sw(τ, μ, mass, px, py)
#     tw_val = tw(τ, μ, mass, px, py)
    
#     mT = mp.sqrt(mass**2 + pT_sq)

#     # First component
#     comp1 = (
#         - (px - 1j * py) * μ / (2 * mass * mT**2 * τ**2 * jw_val)
#         + (px - 1j * py) * β * μ * fhw_val / (2 * mass * τ * eig_val * jw_val)
#         - (px - 1j * py) * SP * (
#             mT**2 * τ**2 * (mL + mass * mT * s * τ * fhw_val) * sw_val
#             + mass * s * (μ + 1j * mT**3 * τ**3 * cjw_val * tw_val)
#         ) / (4 * mass * mT**3 * s * τ**2 * eig_val * jw_val)
#     )

#     # Second component
#     comp2 = (
#         β * (mass * (mass**2 - mL**2 - mT**2) * s + mL * mT**3 * τ * fhw_val)
#     ) / (
#         2 * mass * mT**2 * s * τ * eig_val * jw_val
#     ) - (
#         mL - mass * mT * s * τ * fhw_val
#     ) / (
#         2 * mass * mT * s * τ * jw_val
#     ) - (
#         SP * (mass * mL - mass**2 * mT * s * τ * fhw_val + pT_sq * s * μ * sw_val)
#     ) / (
#         4 * mass * mT**2 * s * τ * eig_val * jw_val
#     )

#     # Third component
#     comp3 = (
#         1/2 + (mL * mT * β) / (2 * mass * s * eig_val) + (
#             SP * (mass**2 * mT * τ * jw_val - 1j * pT_sq * μ * tw_val)
#         ) / (
#             4 * mass * mT**2 * τ * eig_val * jw_val
#         )
#     )

#     # Fourth component
#     comp4 = (
#         (px - 1j * py) * β * μ / (2 * mass * τ * eig_val)
#         - (px - 1j * py) * SP * (
#             mass * mT * s * τ * jw_val * sw_val
#             + 1j * (mL - mass * mT * s * τ * fhw_val) * tw_val
#         ) / (4 * mass * mT * s * eig_val * jw_val)
#     )

#     vec = [
#         mp.mpc(comp1),
#         mp.mpc(comp2),
#         mp.mpc(comp3),
#         mp.mpc(comp4)
#     ]

#     norm = mp.sqrt(sum(vi * mp.conj(vi) for vi in vec))
#     normalized_vec = [vi / norm for vi in vec]

#     return normalized_vec

# def compute_normalized_vector_minusminus(τ, μ, mass, px, py, β, SP):
#     """
#         returns the eigenvector corresponding to -E_-. 
        
#         !!! WARNING: this only works if px^2+py^2 != 0 !!!
#             if px^2 + py^2 == 0 it is not an eigenvector
#     """
#     s = mp.sign(SP)
#     pT_sq = px**2 + py**2
#     mL = mp.sqrt(mass**2 + μ**2 / τ**2)

#     eig_val = eigenvals(τ, μ, 1, -1, β, SP, mass, mp.sqrt(pT_sq))

#     fhw_val = fhw(τ, μ, mass, px, py)
#     jw_val = jw(τ, μ, mass, px, py)
#     cjw_val = mp.conj(jw_val)
#     sw_val = sw(τ, μ, mass, px, py)
#     tw_val = tw(τ, μ, mass, px, py)

#     mT = mp.sqrt(mass**2 + pT_sq)

#     # First component
#     comp1 = (
#         - (px - 1j * py) * μ / (2 * mass * mT**2 * τ**2 * jw_val)
#         + (px - 1j * py) * β * μ * fhw_val / (2 * mass * τ * eig_val * jw_val)
#         - (px - 1j * py) * SP * (
#             mT**2 * τ**2 * (-mL + mass * mT * s * τ * fhw_val) * sw_val
#             + mass * s * (μ + 1j * mT**3 * τ**3 * cjw_val * tw_val)
#         ) / (4 * mass * mT**3 * s * τ**2 * eig_val * jw_val)
#     )

#     # Second component
#     comp2 = (
#         β * (mass * (mass**2 - mL**2 - mT**2) * s - mL * mT**3 * τ * fhw_val)
#     ) / (
#         2 * mass * mT**2 * s * τ * eig_val * jw_val
#     ) + (
#         mL + mass * mT * s * τ * fhw_val
#     ) / (
#         2 * mass * mT * s * τ * jw_val
#     ) + (
#         SP * (mass * mL + mass**2 * mT * s * τ * fhw_val - pT_sq * s * μ * sw_val)
#     ) / (
#         4 * mass * mT**2 * s * τ * eig_val * jw_val
#     )

#     # Third component
#     comp3 = (
#         1/2 - (mL * mT * β) / (2 * mass * s * eig_val) + (
#             SP * (mass**2 * mT * τ * jw_val - 1j * pT_sq * μ * tw_val)
#         ) / (
#             4 * mass * mT**2 * τ * eig_val * jw_val
#         )
#     )

#     # Fourth component
#     comp4 = (
#         (px - 1j * py) * β * μ / (2 * mass * τ * eig_val)
#         - (px - 1j * py) * SP * (
#             mass * mT * s * τ * jw_val * sw_val
#             - 1j * (mL + mass * mT * s * τ * fhw_val) * tw_val
#         ) / (4 * mass * mT * s * eig_val * jw_val)
#     )

#     vec = [
#         mp.mpc(comp1),
#         mp.mpc(comp2),
#         mp.mpc(comp3),
#         mp.mpc(comp4)
#     ]

#     norm = mp.sqrt(sum(vi * mp.conj(vi) for vi in vec))
#     normalized_vec = [vi / norm for vi in vec]

#     return normalized_vec

def compute_normalized_eigenvectors(τ, μ, mass, px, py, β, SP):
    """
        returns the eigenvectors corresponding to (+E_+,+E_-,-E_+,-E_-). 
        
        !!! WARNING: this only works if px^2+py^2 != 0 !!!
            if px^2 + py^2 == 0 it is not an eigenvector
    """
    s=mp.sign(SP)
    pT_sq = px**2 + py**2
    mL = mp.sqrt(mass**2+μ**2/τ**2)
    eps_ = 0#1e-12

    eig_valp = eigenvals(τ, μ, 1, 1, β, SP, mass, mp.sqrt(pT_sq))
    eig_valm = eigenvals(τ, μ, 1, -1, β, SP, mass, mp.sqrt(pT_sq))

    fhw_val = fhw(τ, μ, mass, px, py)#+eps_
    jw_val = jw(τ, μ, mass, px, py)#+eps_
    cjw_val = mp.conj(jw_val)#+eps_
    sw_val = sw(τ, μ, mass, px, py)#+eps_
    tw_val = tw(τ, μ, mass, px, py)#+eps_

    mT = mp.sqrt(mass**2 + pT_sq)

    #################   vpp
    # First component
    comp1 = (
        β * (mass * (-mass**2 + mL**2 + mT**2) * s + mL * mT**3 * τ * fhw_val) 
    ) / (
        2 * mass * mT**2 * s * τ * eig_valp * jw_val
    ) + (
        mL + mass * mT * s * τ * fhw_val
    ) / (
        2 * mass * mT * s * τ * jw_val
    ) + (
        SP * (mass * mL + mass**2 * mT * s * τ * fhw_val - pT_sq * s * μ * sw_val)
    ) / (
        4 * mass * mT**2 * s * τ * eig_valp * jw_val
    )

    # Second component
    comp2 = (
        (px + 1j * py) * μ / (2 * mass * mT**2 * τ**2 * jw_val)
        + (px + 1j * py) * β * μ * fhw_val / (2 * mass * τ * eig_valp * jw_val)
        + (px + 1j * py) * SP * (
            mT**2 * τ**2 * (-mL + mass * mT * s * τ * fhw_val) * sw_val
            + mass * s * (μ + 1j * mT**3 * τ**3 * cjw_val * tw_val)
        ) / (4 * mass * mT**3 * s * τ**2 * eig_valp * jw_val)
    )

    # Third component
    comp3 = (
        (px + 1j * py) * β * μ
    ) / (
        2 * mass * τ * eig_valp
    ) + (
        (px + 1j * py) * SP * (mass * mT * s * τ * jw_val * sw_val - 1j * (mL + mass * mT * s * τ * fhw_val) * tw_val)
    ) / (
        4 * mass * mT * s * eig_valp * jw_val
    )

    # Fourth component
    comp4 = (
        1/2 + (mL * mT * β) / (2 * mass * s * eig_valp) + (
            SP * (mass**2 * mT * τ * jw_val - 1j * pT_sq * μ * tw_val)
        ) / (
            4 * mass * mT**2 * τ * eig_valp * jw_val
        )
    )

    vec = [
        mp.mpc(comp1),
        mp.mpc(comp2),
        mp.mpc(comp3),
        mp.mpc(comp4)
    ]

    norm = mp.sqrt(sum(vi * mp.conj(vi) for vi in vec))
    vpp = [vi / norm for vi in vec]

    #################   vpm
    # First component
    comp1 = (
        β * (mass * (-mass**2 + mL**2 + mT**2) * s - mL * mT**3 * τ * fhw_val)
    ) / (
        2 * mass * mT**2 * s * τ * eig_valm * jw_val
    ) - (
        mL - mass * mT * s * τ * fhw_val
    ) / (
        2 * mass * mT * s * τ * jw_val
    ) - (
        SP * (mass * mL - mass**2 * mT * s * τ * fhw_val + pT_sq * s * μ * sw_val)
    ) / (
        4 * mass * mT**2 * s * τ * eig_valm * jw_val
    )

    # Second component
    comp2 = (
        (px + 1j * py) * μ / (2 * mass * mT**2 * τ**2 * jw_val)
        + (px + 1j * py) * β * μ * fhw_val / (2 * mass * τ * eig_valm * jw_val)
        + (px + 1j * py) * SP * (
            mT**2 * τ**2 * (mL + mass * mT * s * τ * fhw_val) * sw_val
            + mass * s * (μ + 1j * mT**3 * τ**3 * cjw_val * tw_val)
        ) / (4 * mass * mT**3 * s * τ**2 * eig_valm * jw_val)
    )

    # Third component
    comp3 = (
        (px + 1j * py) * β * μ
    ) / (
        2 * mass * τ * eig_valm
    ) + (
        (px + 1j * py) * SP * (mass * mT * s * τ * jw_val * sw_val + 1j * (mL - mass * mT * s * τ * fhw_val) * tw_val)
    ) / (
        4 * mass * mT * s * eig_valm * jw_val
    )

    # Fourth component
    comp4 = (
        1/2 - (mL * mT * β) / (2 * mass * s * eig_valm) + (
            SP * (mass**2 * mT * τ * jw_val - 1j * pT_sq * μ * tw_val)
        ) / (
            4 * mass * mT**2 * τ * eig_valm * jw_val
        )
    )

    vec = [
        mp.mpc(comp1),
        mp.mpc(comp2),
        mp.mpc(comp3),
        mp.mpc(comp4)
    ]

    norm = mp.sqrt(sum(vi * mp.conj(vi) for vi in vec))
    vpm = [vi / norm for vi in vec]

    #################   vmp
    # First component
    comp1 = (
        - (px - 1j * py) * μ / (2 * mass * mT**2 * τ**2 * jw_val)
        + (px - 1j * py) * β * μ * fhw_val / (2 * mass * τ * eig_valp * jw_val)
        - (px - 1j * py) * SP * (
            mT**2 * τ**2 * (mL + mass * mT * s * τ * fhw_val) * sw_val
            + mass * s * (μ + 1j * mT**3 * τ**3 * cjw_val * tw_val)
        ) / (4 * mass * mT**3 * s * τ**2 * eig_valp * jw_val)
    )

    # Second component
    comp2 = (
        β * (mass * (mass**2 - mL**2 - mT**2) * s + mL * mT**3 * τ * fhw_val)
    ) / (
        2 * mass * mT**2 * s * τ * eig_valp * jw_val
    ) - (
        mL - mass * mT * s * τ * fhw_val
    ) / (
        2 * mass * mT * s * τ * jw_val
    ) - (
        SP * (mass * mL - mass**2 * mT * s * τ * fhw_val + pT_sq * s * μ * sw_val)
    ) / (
        4 * mass * mT**2 * s * τ * eig_valp * jw_val
    )

    # Third component
    comp3 = (
        1/2 + (mL * mT * β) / (2 * mass * s * eig_valp) + (
            SP * (mass**2 * mT * τ * jw_val - 1j * pT_sq * μ * tw_val)
        ) / (
            4 * mass * mT**2 * τ * eig_valp * jw_val
        )
    )

    # Fourth component
    comp4 = (
        (px - 1j * py) * β * μ / (2 * mass * τ * eig_valp)
        - (px - 1j * py) * SP * (
            mass * mT * s * τ * jw_val * sw_val
            + 1j * (mL - mass * mT * s * τ * fhw_val) * tw_val
        ) / (4 * mass * mT * s * eig_valp * jw_val)
    )

    vec = [
        mp.mpc(comp1),
        mp.mpc(comp2),
        mp.mpc(comp3),
        mp.mpc(comp4)
    ]

    norm = mp.sqrt(sum(vi * mp.conj(vi) for vi in vec))
    vmp = [vi / norm for vi in vec]

    #################   vmm
    # First component
    comp1 = (
        - (px - 1j * py) * μ / (2 * mass * mT**2 * τ**2 * jw_val)
        + (px - 1j * py) * β * μ * fhw_val / (2 * mass * τ * eig_valm * jw_val)
        - (px - 1j * py) * SP * (
            mT**2 * τ**2 * (-mL + mass * mT * s * τ * fhw_val) * sw_val
            + mass * s * (μ + 1j * mT**3 * τ**3 * cjw_val * tw_val)
        ) / (4 * mass * mT**3 * s * τ**2 * eig_valm * jw_val)
    )

    # Second component
    comp2 = (
        β * (mass * (mass**2 - mL**2 - mT**2) * s - mL * mT**3 * τ * fhw_val)
    ) / (
        2 * mass * mT**2 * s * τ * eig_valm * jw_val
    ) + (
        mL + mass * mT * s * τ * fhw_val
    ) / (
        2 * mass * mT * s * τ * jw_val
    ) + (
        SP * (mass * mL + mass**2 * mT * s * τ * fhw_val - pT_sq * s * μ * sw_val)
    ) / (
        4 * mass * mT**2 * s * τ * eig_valm * jw_val
    )

    # Third component
    comp3 = (
        1/2 - (mL * mT * β) / (2 * mass * s * eig_valm) + (
            SP * (mass**2 * mT * τ * jw_val - 1j * pT_sq * μ * tw_val)
        ) / (
            4 * mass * mT**2 * τ * eig_valm * jw_val
        )
    )

    # Fourth component
    comp4 = (
        (px - 1j * py) * β * μ / (2 * mass * τ * eig_valm)
        - (px - 1j * py) * SP * (
            mass * mT * s * τ * jw_val * sw_val
            - 1j * (mL + mass * mT * s * τ * fhw_val) * tw_val
        ) / (4 * mass * mT * s * eig_valm * jw_val)
    )

    vec = [
        mp.mpc(comp1),
        mp.mpc(comp2),
        mp.mpc(comp3),
        mp.mpc(comp4)
    ]

    norm = mp.sqrt(sum(vi * mp.conj(vi) for vi in vec))
    vmm = [vi / norm for vi in vec]

    return vpp,vpm,vmp,vmm

def NOFUNCTIONScompute_normalized_eigenvectors(τ, μ, mass, px, py, β, SP,fhw_val,jw_val,sw_val,tw_val):
    """
        returns the eigenvectors corresponding to (+E_+,+E_-,-E_+,-E_-). 
        
        !!! WARNING: this only works if px^2+py^2 != 0 !!!
            if px^2 + py^2 == 0 it is not an eigenvector
    """
    s=mp.sign(SP)
    pT_sq = px**2 + py**2
    mL = mp.sqrt(mass**2+μ**2/τ**2)
    eps_ = 0#1e-12

    eig_valp = eigenvals(τ, μ, 1, 1, β, SP, mass, mp.sqrt(pT_sq))
    eig_valm = eigenvals(τ, μ, 1, -1, β, SP, mass, mp.sqrt(pT_sq))

    cjw_val = mp.conj(jw_val)#+eps_
    
    mT = mp.sqrt(mass**2 + pT_sq)

    #################   vpp
    # First component
    comp1 = (
        β * (mass * (-mass**2 + mL**2 + mT**2) * s + mL * mT**3 * τ * fhw_val) 
    ) / (
        2 * mass * mT**2 * s * τ * eig_valp * jw_val
    ) + (
        mL + mass * mT * s * τ * fhw_val
    ) / (
        2 * mass * mT * s * τ * jw_val
    ) + (
        SP * (mass * mL + mass**2 * mT * s * τ * fhw_val - pT_sq * s * μ * sw_val)
    ) / (
        4 * mass * mT**2 * s * τ * eig_valp * jw_val
    )

    # Second component
    comp2 = (
        (px + 1j * py) * μ / (2 * mass * mT**2 * τ**2 * jw_val)
        + (px + 1j * py) * β * μ * fhw_val / (2 * mass * τ * eig_valp * jw_val)
        + (px + 1j * py) * SP * (
            mT**2 * τ**2 * (-mL + mass * mT * s * τ * fhw_val) * sw_val
            + mass * s * (μ + 1j * mT**3 * τ**3 * cjw_val * tw_val)
        ) / (4 * mass * mT**3 * s * τ**2 * eig_valp * jw_val)
    )

    # Third component
    comp3 = (
        (px + 1j * py) * β * μ
    ) / (
        2 * mass * τ * eig_valp
    ) + (
        (px + 1j * py) * SP * (mass * mT * s * τ * jw_val * sw_val - 1j * (mL + mass * mT * s * τ * fhw_val) * tw_val)
    ) / (
        4 * mass * mT * s * eig_valp * jw_val
    )

    # Fourth component
    comp4 = (
        1/2 + (mL * mT * β) / (2 * mass * s * eig_valp) + (
            SP * (mass**2 * mT * τ * jw_val - 1j * pT_sq * μ * tw_val)
        ) / (
            4 * mass * mT**2 * τ * eig_valp * jw_val
        )
    )

    vec = [
        mp.mpc(comp1),
        mp.mpc(comp2),
        mp.mpc(comp3),
        mp.mpc(comp4)
    ]

    norm = mp.sqrt(sum(vi * mp.conj(vi) for vi in vec))
    vpp = [vi / norm for vi in vec]

    #################   vpm
    # First component
    comp1 = (
        β * (mass * (-mass**2 + mL**2 + mT**2) * s - mL * mT**3 * τ * fhw_val)
    ) / (
        2 * mass * mT**2 * s * τ * eig_valm * jw_val
    ) - (
        mL - mass * mT * s * τ * fhw_val
    ) / (
        2 * mass * mT * s * τ * jw_val
    ) - (
        SP * (mass * mL - mass**2 * mT * s * τ * fhw_val + pT_sq * s * μ * sw_val)
    ) / (
        4 * mass * mT**2 * s * τ * eig_valm * jw_val
    )

    # Second component
    comp2 = (
        (px + 1j * py) * μ / (2 * mass * mT**2 * τ**2 * jw_val)
        + (px + 1j * py) * β * μ * fhw_val / (2 * mass * τ * eig_valm * jw_val)
        + (px + 1j * py) * SP * (
            mT**2 * τ**2 * (mL + mass * mT * s * τ * fhw_val) * sw_val
            + mass * s * (μ + 1j * mT**3 * τ**3 * cjw_val * tw_val)
        ) / (4 * mass * mT**3 * s * τ**2 * eig_valm * jw_val)
    )

    # Third component
    comp3 = (
        (px + 1j * py) * β * μ
    ) / (
        2 * mass * τ * eig_valm
    ) + (
        (px + 1j * py) * SP * (mass * mT * s * τ * jw_val * sw_val + 1j * (mL - mass * mT * s * τ * fhw_val) * tw_val)
    ) / (
        4 * mass * mT * s * eig_valm * jw_val
    )

    # Fourth component
    comp4 = (
        1/2 - (mL * mT * β) / (2 * mass * s * eig_valm) + (
            SP * (mass**2 * mT * τ * jw_val - 1j * pT_sq * μ * tw_val)
        ) / (
            4 * mass * mT**2 * τ * eig_valm * jw_val
        )
    )

    vec = [
        mp.mpc(comp1),
        mp.mpc(comp2),
        mp.mpc(comp3),
        mp.mpc(comp4)
    ]

    norm = mp.sqrt(sum(vi * mp.conj(vi) for vi in vec))
    vpm = [vi / norm for vi in vec]

    #################   vmp
    # First component
    comp1 = (
        - (px - 1j * py) * μ / (2 * mass * mT**2 * τ**2 * jw_val)
        + (px - 1j * py) * β * μ * fhw_val / (2 * mass * τ * eig_valp * jw_val)
        - (px - 1j * py) * SP * (
            mT**2 * τ**2 * (mL + mass * mT * s * τ * fhw_val) * sw_val
            + mass * s * (μ + 1j * mT**3 * τ**3 * cjw_val * tw_val)
        ) / (4 * mass * mT**3 * s * τ**2 * eig_valp * jw_val)
    )

    # Second component
    comp2 = (
        β * (mass * (mass**2 - mL**2 - mT**2) * s + mL * mT**3 * τ * fhw_val)
    ) / (
        2 * mass * mT**2 * s * τ * eig_valp * jw_val
    ) - (
        mL - mass * mT * s * τ * fhw_val
    ) / (
        2 * mass * mT * s * τ * jw_val
    ) - (
        SP * (mass * mL - mass**2 * mT * s * τ * fhw_val + pT_sq * s * μ * sw_val)
    ) / (
        4 * mass * mT**2 * s * τ * eig_valp * jw_val
    )

    # Third component
    comp3 = (
        1/2 + (mL * mT * β) / (2 * mass * s * eig_valp) + (
            SP * (mass**2 * mT * τ * jw_val - 1j * pT_sq * μ * tw_val)
        ) / (
            4 * mass * mT**2 * τ * eig_valp * jw_val
        )
    )

    # Fourth component
    comp4 = (
        (px - 1j * py) * β * μ / (2 * mass * τ * eig_valp)
        - (px - 1j * py) * SP * (
            mass * mT * s * τ * jw_val * sw_val
            + 1j * (mL - mass * mT * s * τ * fhw_val) * tw_val
        ) / (4 * mass * mT * s * eig_valp * jw_val)
    )

    vec = [
        mp.mpc(comp1),
        mp.mpc(comp2),
        mp.mpc(comp3),
        mp.mpc(comp4)
    ]

    norm = mp.sqrt(sum(vi * mp.conj(vi) for vi in vec))
    vmp = [vi / norm for vi in vec]

    #################   vmm
    # First component
    comp1 = (
        - (px - 1j * py) * μ / (2 * mass * mT**2 * τ**2 * jw_val)
        + (px - 1j * py) * β * μ * fhw_val / (2 * mass * τ * eig_valm * jw_val)
        - (px - 1j * py) * SP * (
            mT**2 * τ**2 * (-mL + mass * mT * s * τ * fhw_val) * sw_val
            + mass * s * (μ + 1j * mT**3 * τ**3 * cjw_val * tw_val)
        ) / (4 * mass * mT**3 * s * τ**2 * eig_valm * jw_val)
    )

    # Second component
    comp2 = (
        β * (mass * (mass**2 - mL**2 - mT**2) * s - mL * mT**3 * τ * fhw_val)
    ) / (
        2 * mass * mT**2 * s * τ * eig_valm * jw_val
    ) + (
        mL + mass * mT * s * τ * fhw_val
    ) / (
        2 * mass * mT * s * τ * jw_val
    ) + (
        SP * (mass * mL + mass**2 * mT * s * τ * fhw_val - pT_sq * s * μ * sw_val)
    ) / (
        4 * mass * mT**2 * s * τ * eig_valm * jw_val
    )

    # Third component
    comp3 = (
        1/2 - (mL * mT * β) / (2 * mass * s * eig_valm) + (
            SP * (mass**2 * mT * τ * jw_val - 1j * pT_sq * μ * tw_val)
        ) / (
            4 * mass * mT**2 * τ * eig_valm * jw_val
        )
    )

    # Fourth component
    comp4 = (
        (px - 1j * py) * β * μ / (2 * mass * τ * eig_valm)
        - (px - 1j * py) * SP * (
            mass * mT * s * τ * jw_val * sw_val
            - 1j * (mL + mass * mT * s * τ * fhw_val) * tw_val
        ) / (4 * mass * mT * s * eig_valm * jw_val)
    )

    vec = [
        mp.mpc(comp1),
        mp.mpc(comp2),
        mp.mpc(comp3),
        mp.mpc(comp4)
    ]

    norm = mp.sqrt(sum(vi * mp.conj(vi) for vi in vec))
    vmm = [vi / norm for vi in vec]

    return vpp,vpm,vmp,vmm

def Umatrix(τ, μ, mass, px, py, β, SP):
    """
        returns the unitary matrix built with the eigenvectors as columns as a numpy array. 
    """
    if (px**2+py**2 == 0):
        print("Error in the function Umatrix: pT^2 is too small. The eigenvectors implemented in the function Umatrix only work for pT^2 != 0.")
    
    v1,v2,v3,v4 = compute_normalized_eigenvectors(τ, μ, mass, px, py, β, SP)
    
    v1 = to_numpy(v1)
    v2 = to_numpy(v2)
    v3 = to_numpy(v3)
    v4 = to_numpy(v4)
    
    return np.column_stack((v1, v2, v3, v4))

def mpmath_Umatrix(τ, μ, mass, px, py, β, SP):
    """
        returns the unitary matrix built with the eigenvectors as columns as a mpmath matrix. 
    """
    if (px**2+py**2 == 0):
        print("Error in the function mpmath_Umatrix: pT^2 is too small. The eigenvectors implemented in the function mpmath_Umatrix only work for pT^2 != 0.")
    
    v1,v2,v3,v4 = compute_normalized_eigenvectors(τ, μ, mass, px, py, β, SP)
    
    U = mp.matrix([[v1[0],v2[0],v3[0],v4[0]],
                   [v1[1],v2[1],v3[1],v4[1]],
                   [v1[2],v2[2],v3[2],v4[2]],
                   [v1[3],v2[3],v3[3],v4[3]]]
                  )
    return U

def U_and_Udagger_matrices(τ, μ, mass, px, py, β, SP):
    """
        returns both the unitary matrix built with the eigenvectors as columns and its hermitian conjugate as a numpy arrays. 
    """
    if (px**2+py**2 == 0):
        print("Error in the function U_and_Udagger_matrices: pT^2 is too small. The eigenvectors implemented in the function U_and_Udagger_matrices only work for pT^2 != 0.")
    
    v1,v2,v3,v4 = compute_normalized_eigenvectors(τ, μ, mass, px, py, β, SP)
    
    v1 = to_numpy(v1)
    v2 = to_numpy(v2)
    v3 = to_numpy(v3)
    v4 = to_numpy(v4)
    
    U = np.column_stack((v1, v2, v3, v4))
    Ud = np.transpose(np.conjugate(U))
    return U,Ud

def mpmath_U_and_Udagger_matrices(τ, μ, mass, px, py, β, SP):
    """
        returns both the unitary matrix built with the eigenvectors as columns and its hermitian conjugate as a mpmath matrices. 
    """
    if (px**2+py**2 == 0):
        print("Error in the function mpmath_U_and_Udagger_matrices: pT^2 is too small. The eigenvectors implemented in the function mpmath_U_and_Udagger_matrices only work for pT^2 != 0.")
    
    v1,v2,v3,v4 = compute_normalized_eigenvectors(τ, μ, mass, px, py, β, SP)
    
    U = mp.matrix([[v1[0],v2[0],v3[0],v4[0]],
                   [v1[1],v2[1],v3[1],v4[1]],
                   [v1[2],v2[2],v3[2],v4[2]],
                   [v1[3],v2[3],v3[3],v4[3]]]
                  )
    Ud = U.transpose_conj()
    
    return U,Ud

def Htot(τ, μ, mass, px, py, β, SP):
    """
        returns the total hamiltonian to be diagonalized. 
        In this function Htot = β*Htherm +SP*Hspin with the density operator being rho ~ exp(-Htot)
    """
    pT_sq = px**2 + py**2
    mT = mp.sqrt(mass**2 + pT_sq)

    frackhw_val = fhw(τ, μ, mass, px, py)
    sw_val = sw(τ, μ, mass, px, py)
    conjtw_val = ctw(τ, μ, mass, px, py)
    jw_val = jw(τ, μ, mass, px, py)
    conjjw_val = cjw(τ, μ, mass, px, py)
    tw_val = tw(τ, μ, mass, px, py)

    matrix = mp.matrix(4, 4)

    # First row
    matrix[0, 0] = (mass * SP) / (2 * mT) + mT**2 * β * τ * frackhw_val
    matrix[0, 1] = 1/2 * (-px + 1j * py) * SP * τ * sw_val
    matrix[0, 2] = 1/2 * (1j * px + py) * SP * τ * conjtw_val
    matrix[0, 3] = mT**2 * β * τ * conjjw_val

    # Second row
    matrix[1, 0] = 1/2 * (-px - 1j * py) * SP * τ * sw_val
    matrix[1, 1] = -(mass * SP) / (2 * mT) + mT**2 * β * τ * frackhw_val
    matrix[1, 2] = mT**2 * β * τ * conjjw_val
    matrix[1, 3] = 1/2 * (1j * px - py) * SP * τ * conjtw_val

    # Third row
    matrix[2, 0] = 1/2 * (-1j * px + py) * SP * τ * tw_val
    matrix[2, 1] = mT**2 * β * τ * jw_val
    matrix[2, 2] = -(mass * SP) / (2 * mT) - mT**2 * β * τ * frackhw_val
    matrix[2, 3] = 1/2 * (px + 1j * py) * SP * τ * sw_val

    # Fourth row
    matrix[3, 0] = mT**2 * β * τ * jw_val
    matrix[3, 1] = 1/2 * (-1j * px - py) * SP * τ * tw_val
    matrix[3, 2] = 1/2 * (px - 1j * py) * SP * τ * sw_val
    matrix[3, 3] = (mass * SP) / (2 * mT) - mT**2 * β * τ * frackhw_val

    return matrix

def fermi_minor_part(τ, μ, mass, px, py, β, SP, ζ=0):
    """
        returns the 2x2 matrix corresponding to quasi-particle number: <alpha^dagger_r alpha_s>
        The ratio of chemical potential over temperature ζ is an optional parameter.
    """
    Ep = eigenvals(τ, μ, 1, 1, β, SP, mass,mp.sqrt(px**2+py**2))-ζ
    Em = eigenvals(τ, μ, 1, -1, β, SP, mass,mp.sqrt(px**2+py**2))-ζ
    return mp.matrix([[1/(mp.exp(Ep)+1),0],
                      [0,1/(mp.exp(Em)+1)]])

def fermi_minor_anti(τ, μ, mass, px, py, β, SP, ζ=0):
    """
        returns the 2x2 matrix corresponding to quasi-antiparticle number: <beta^dagger_r beta_s>
        The ratio of chemical potential over temperature ζ is an optional parameter.
    """
    Ep = eigenvals(τ, μ, 1, 1, β, SP, mass,mp.sqrt(px**2+py**2))+ζ
    Em = eigenvals(τ, μ, 1, -1, β, SP, mass,mp.sqrt(px**2+py**2))+ζ
    return mp.matrix([[1/(mp.exp(Ep)+1),0],
                        [0,1/(mp.exp(Em)+1)]])
    
    
#####################       BOGOLIUBOV MANIPULATIONS      ########################

    
def block_bogoliubov(τ, μ, mass, px, py, β, SP):
    """
        returns the blocks of the matrix U ={{u,v},{w,z}} as mpmath matrices.
        The blocks are used to transform fermion operators (A,B^dagger)^T = U (alpha, beta^dagger)^T
    """
    
    U = mpmath_Umatrix(τ, μ, mass, px, py, β, SP)
     
    u = mp.matrix([[U[0,0],U[0,1]],
                   [U[1,0],U[1,1]]])
    v = mp.matrix([[U[0,2],U[0,3]],
                   [U[1,2],U[1,3]]])
    w = mp.matrix([[U[2,0],U[2,1]],
                   [U[3,0],U[3,1]]])
    z = mp.matrix([[U[2,2],U[2,3]],
                   [U[3,2],U[3,3]]])
    return u,v,w,z

def block_and_dagger_bogoliubov(τ, μ, mass, px, py, β, SP):
    """
        returns  both the blocks of the matrix U ={{u,v},{w,z}} and their hermitian conjugate (the blocks of U^dagger) as mpmath matrices.
        The blocks are used to transform fermion operators (A,B^dagger)^T = U (alpha, beta^dagger)^T
    """
    
    U = mpmath_Umatrix(τ, μ, mass, px, py, β, SP)
     
    u = mp.matrix([[U[0,0],U[0,1]],
                   [U[1,0],U[1,1]]])
    v = mp.matrix([[U[0,2],U[0,3]],
                   [U[1,2],U[1,3]]])
    w = mp.matrix([[U[2,0],U[2,1]],
                   [U[3,0],U[3,1]]])
    z = mp.matrix([[U[2,2],U[2,3]],
                   [U[3,2],U[3,3]]])
    
    ud = u.transpose_conj()
    vd = v.transpose_conj()
    wd = w.transpose_conj()
    zd = z.transpose_conj()
    
    return u,v,w,z,ud,vd,wd,zd    

def numpy_block_and_dagger_bogoliubov(τ, μ, mass, px, py, β, SP):
    """
        returns  both the blocks of the matrix U ={{u,v},{w,z}} and their hermitian conjugate (the blocks of U^dagger) as mpmath matrices.
        The blocks are used to transform fermion operators (A,B^dagger)^T = U (alpha, beta^dagger)^T
    """
    
    U = Umatrix(τ, μ, mass, px, py, β, SP)
     
    u = np.array([[U[0,0],U[0,1]],
                   [U[1,0],U[1,1]]])
    v = np.array([[U[0,2],U[0,3]],
                   [U[1,2],U[1,3]]])
    w = np.array([[U[2,0],U[2,1]],
                   [U[3,0],U[3,1]]])
    z = np.array([[U[2,2],U[2,3]],
                   [U[3,2],U[3,3]]])
    
    ud = u.conj().T
    vd = v.conj().T
    wd = w.conj().T
    zd = z.conj().T
    
    return u,v,w,z,ud,vd,wd,zd   

def Odag_O(τ, μ, mass, px, py, β, SP, mu_T=0):
    """
        returns <A^dagger_r A_s>, <B^dagger_r B_s>, <A^dagger_r B^dagger_s>, and <B_r A_s> as mpmatrices
        r and s are row and column of the matrix 
    """
    u,v,w,z,ud,vd,wd,zd = block_and_dagger_bogoliubov(τ, μ, mass, px, py, β, SP)
    
    alphadag_alpha = fermi_minor_part(τ, μ, mass, px, py, β, SP, mu_T)
    betadag_beta = fermi_minor_anti(τ, μ, mass, px, py, β, SP, mu_T)
    
    Adag_A_T = u*alphadag_alpha*ud + v*vd - v*betadag_beta*vd 
    Bdag_B = z*betadag_beta*zd+w*wd-w*alphadag_alpha*wd  
    Adag_Bdag_T = w*alphadag_alpha*ud + z*vd - z*betadag_beta*vd
    B_A = -Adag_Bdag_T.conjugate() #########################################################àbewere of the sign
    
    return Adag_A_T.transpose(), Bdag_B, B_A, Adag_Bdag_T.transpose()

def numpy_Odag_O(τ, μ, mass, px, py, β, SP, mu_T=0):
    """
        returns <A^dagger_r A_s>, <B^dagger_r B_s>, <A^dagger_r B^dagger_s>, and <B_r A_s> as numpy arrays
        r and s are row and column of the matrix 
    """
    u,v,w,z,ud,vd,wd,zd = numpy_block_and_dagger_bogoliubov(τ, μ, mass, px, py, β, SP)
    
    alphadag_alpha = to_numpy(fermi_minor_part(τ, μ, mass, px, py, β, SP, mu_T))
    betadag_beta = to_numpy(fermi_minor_anti(τ, μ, mass, px, py, β, SP, mu_T))
    
    Adag_A_T = np.dot(u,np.dot(alphadag_alpha,ud)) + np.dot(v,vd) - np.dot(v,np.dot(betadag_beta,vd)) 
    Bdag_B = np.dot(z,np.dot(betadag_beta,zd))+np.dot(w,wd)-np.dot(w,np.dot(alphadag_alpha,wd))  
    Adag_Bdag_T = np.dot(w,np.dot(alphadag_alpha,ud)) + np.dot(z,vd) - np.dot(z,np.dot(betadag_beta,vd))
    B_A = -Adag_Bdag_T.conj()###########################################################BEWERE of the sign
    
    return Adag_A_T.T, Bdag_B, B_A, Adag_Bdag_T.T
    
def Adag_A(τ, μ, mass, px, py, β, SP, mu_T=0):
    """
        returns <A^dagger_r A_s> as mpmatrix
        r and s are row and column of the matrix 
    """
    U = mpmath_Umatrix(τ, μ, mass, px, py, β, SP)
     
    u = mp.matrix([[U[0,0],U[0,1]],
                   [U[1,0],U[1,1]]])
    v = mp.matrix([[U[0,2],U[0,3]],
                   [U[1,2],U[1,3]]])
    ud = u.transpose_conj()
    vd = v.transpose_conj()
    
    alphadag_alpha = fermi_minor_part(τ, μ, mass, px, py, β, SP, mu_T)
    betadag_beta = fermi_minor_anti(τ, μ, mass, px, py, β, SP, mu_T)
    
    Adag_A_T = u*alphadag_alpha*ud + v*vd - v*betadag_beta*vd 
    
    return Adag_A_T.transpose()
    
def numpy_Adag_A(τ, μ, mass, px, py, β, SP, mu_T=0):
    """
        returns <A^dagger_r A_s> as mpmatrix
        r and s are row and column of the matrix 
    """
    U = Umatrix(τ, μ, mass, px, py, β, SP)
    
    u = np.array([[U[0,0],U[0,1]],
                [U[1,0],U[1,1]]])
    v = np.array([[U[0,2],U[0,3]],
                [U[1,2],U[1,3]]])
    ud = u.conj().T
    vd = v.conj().T
    
    alphadag_alpha = to_numpy(fermi_minor_part(τ, μ, mass, px, py, β, SP, mu_T))
    betadag_beta = to_numpy(fermi_minor_anti(τ, μ, mass, px, py, β, SP, mu_T))
    
    Adag_A_T = np.dot(u,np.dot(alphadag_alpha,ud)) + np.dot(v,vd) - np.dot(v,np.dot(betadag_beta,vd)) 
    
    return Adag_A_T.T


def tabulating_canonical(mass, px, py, μ, τ, β, SP, mu_T=0, precision = 50):
    if(px**2+py**2<(1e-5)**2):
        px=1e-5
        py=1e-5
    
    if(μ**2 < (1e-5)**2):
        μ=1e-5
    
    mass = mp.mpf(mass)
    px   = mp.mpf(px)
    py   = mp.mpf(py)
    μ    = mp.mpf(μ)
    τ    = mp.mpf(τ)
    β    = mp.mpf(β)
    SP   = mp.mpf(SP)
    mu_T = mp.mpf(mu_T)
    
    mT2 = mass**2+px**2+py**2
    
    with mp.workprec(precision):
        fhw_val, jw_val, sw_val, tw_val, fw_val, ww_val, zw_val = FAST_specialfunctions(τ, μ, mass, px, py)
        

    v1,v2,v3,v4 = NOFUNCTIONScompute_normalized_eigenvectors(τ, μ, mass, px, py, β, SP,fhw_val, jw_val, sw_val,tw_val)
    
    U = mp.matrix([[v1[0],v2[0],v3[0],v4[0]],
                   [v1[1],v2[1],v3[1],v4[1]],
                   [v1[2],v2[2],v3[2],v4[2]],
                   [v1[3],v2[3],v3[3],v4[3]]]
                  )
    u = mp.matrix([[U[0,0],U[0,1]],
                   [U[1,0],U[1,1]]])
    v = mp.matrix([[U[0,2],U[0,3]],
                   [U[1,2],U[1,3]]])
    w = mp.matrix([[U[2,0],U[2,1]],
                   [U[3,0],U[3,1]]])
    z = mp.matrix([[U[2,2],U[2,3]],
                   [U[3,2],U[3,3]]])
    
    ud = u.transpose_conj()
    vd = v.transpose_conj()
    wd = w.transpose_conj()
    zd = z.transpose_conj()
    
    alphadag_alpha = fermi_minor_part(τ, μ, mass, px, py, β, SP, mu_T)
    betadag_beta = fermi_minor_anti(τ, μ, mass, px, py, β, SP, mu_T)
    
    Adag_A_T = u*alphadag_alpha*ud  - v*betadag_beta*vd 
    Bdag_B = z*betadag_beta*zd-w*alphadag_alpha*wd  
    Adag_Bdag_T = w*alphadag_alpha*ud  - z*betadag_beta*vd
    
    energy_density = (2*mp.pi)**(-3) *(mT2)*(fhw_val* (trace(Adag_A_T)+trace(Bdag_B))+2*mp.re(mp.conj(jw_val)*trace(pauli_matrices(1)*Adag_Bdag_T)))
    long_pressure = -(2*mp.pi)**(-3)*(mp.sqrt(mT2)*μ/τ)*(sw_val* (trace(Adag_A_T)+trace(Bdag_B))+2*mp.im(mp.conj(tw_val)*trace(pauli_matrices(1)*Adag_Bdag_T)))                                                     
    
    
    pre = (1/2)*(2*mp.pi)**(-3)
    block_A = 2*px*pauli_matrices(0)*mp.im(fw_val)+2*mp.re(fw_val)*(pauli_matrices(2)*((mass**2+py**2+mass*mp.sqrt(mT2))/(mass+mp.sqrt(mT2)))+pauli_matrices(1)*((px*py)/(mass+mp.sqrt(mT2))))
    block_B = 2*px*pauli_matrices(0)*mp.im(fw_val)+2*mp.re(fw_val)*(-pauli_matrices(2)*((mass**2+py**2+mass*mp.sqrt(mT2))/(mass+mp.sqrt(mT2)))+pauli_matrices(1)*((px*py)/(mass+mp.sqrt(mT2))))
    block_C =-px*zw_val*pauli_matrices(1)+ww_val*(pauli_matrices(3)*((mass**2+py**2+mass*mp.sqrt(mT2))/(mass+mp.sqrt(mT2)))+1j*pauli_matrices(0)*((px*py)/(mass+mp.sqrt(mT2))))

    integrandx = pre*px*(trace(Adag_A_T*block_A)+trace(Bdag_B*block_B)+2*mp.re(trace(block_C*Adag_Bdag_T)))

    block_Ay = 2*py*pauli_matrices(0)*mp.im(fw_val)-2*mp.re(fw_val)*(pauli_matrices(1)*((mass**2+px**2+mass*mp.sqrt(mT2))/(mass+mp.sqrt(mT2)))+pauli_matrices(2)*((px*py)/(mass+mp.sqrt(mT2))))
    block_By = 2*py*pauli_matrices(0)*mp.im(fw_val)-2*mp.re(fw_val)*(pauli_matrices(1)*((mass**2+px**2+mass*mp.sqrt(mT2))/(mass+mp.sqrt(mT2)))-pauli_matrices(2)*((px*py)/(mass+mp.sqrt(mT2))))
    block_Cy =-py*zw_val*pauli_matrices(1)-ww_val*(pauli_matrices(0)*(1j*(mass**2+px**2+mass*mp.sqrt(mT2))/(mass+mp.sqrt(mT2)))+pauli_matrices(3)*((px*py)/(mass+mp.sqrt(mT2))))

    integrandy = pre*py*(trace(Adag_A_T*block_Ay)+trace(Bdag_B*block_By)+2*mp.re(trace(block_Cy*Adag_Bdag_T)))
    
    transv_pressure = integrandx+integrandy
    
    return energy_density, transv_pressure, long_pressure


def tabulating_belinfante(mass, px, py, μ, τ, β, precision = 50):
    with mp.workprec(precision):
        energy_density = (2/ τ) *(2*mp.pi)**(-3)*mp.sqrt(mass**2+px**2+py**2+(μ/τ)**2)*2/(mp.exp(β*mp.sqrt(mass**2+px**2+py**2+(μ/τ)**2))+1)
        long_pressure = (2/ τ) *(2*mp.pi)**(-3)*( μ**2 /(τ**2 *mp.sqrt(mass**2+px**2+py**2+(μ/τ)**2)))*2/(mp.exp(β*mp.sqrt(mass**2+px**2+py**2+(μ/τ)**2))+1)
        transv_pressure = (2/ τ) *(2*mp.pi)**(-3)*( (px**2+py**2) /(2 *mp.sqrt(mass**2+px**2+py**2+(μ/τ)**2)))*2/(mp.exp(β*mp.sqrt(mass**2+px**2+py**2+(μ/τ)**2))+1)
         
    
    return energy_density,transv_pressure,long_pressure
    


#####################       UTILS      ########################


def to_numpy(mpobject):
    """
        Converts a mpmath float, array or matrix to a numpy number or array
    """
    if isinstance(mpobject,(mp.mpf,mp.mpc)):
        return np.complex128(float(mpobject.real)+1j*float(mpobject.imag))
        
    elif isinstance(mpobject,list):
        v = []
        for o in mpobject:
            v.append(np.complex128(float(o.real)+1j*float(o.imag)))
        
        return np.array(v)
    
    elif isinstance(mpobject,mp.matrix):
        rows = mpobject.rows
        cols = mpobject.cols
        v = []
        for row in range(rows):
            w = []
            for col in range(cols):
                element = mpobject[row, col]
                w.append(np.complex128(float(element.real)+1j*float(element.imag)))
            v.append(w)
        return np.array(v)
        
    else:
        print("to_numpy function error! Unkonwn case of mpobject conversion.")
        return np.nan

    
def pauli_matrices(i):
    """
        returns the ith component of the "four-vector" of Pauli matrices [I,sx,sy,sz]_i
    """
    vec_Pauli = [
        [[1,0],[0,1]],
        [[0,1],[1,0]],
        [[0,-1j],[1j,0]],
        [[1,0],[0,-1]]
     ]
    return mp.matrix(vec_Pauli[i])
       
def trace(A):
    """
        returns the trace of a mpmath matrix
    """
    return sum(A[i, i] for i in range(min(A.rows, A.cols)))       