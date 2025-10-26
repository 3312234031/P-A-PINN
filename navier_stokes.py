
from __future__ import annotations

from typing import Optional
from typing import Tuple
from typing import Union
import paddle
import sympy as sp
from sympy.parsing import sympy_parser as sp_parser

from ppsci.equation.pde import base


class NavierStokes(base.PDE):
    

    def __init__(
        self,
        nu: Union[float, str],
        rho: Union[float, str],
        dim: int,
        time: bool,
        k: Union[float, str] = 0.1,  
        epsilon: Union[float, str] = 0.001,  
        C_mu: float = 0.09,  
        C_epsilon_1: float = 1.44,  
        C_epsilon_2: float = 1.92,  
        detach_keys: Optional[Tuple[str, ...]] = None,
    ):
        super().__init__()
        self.detach_keys = detach_keys
        self.dim = dim
        self.time = time
        self.k = k        
        self.epsilon = epsilon  
        self.C_mu = C_mu  
        self.C_epsilon_1 = C_epsilon_1  
        self.C_epsilon_2 = C_epsilon_2  
        
        
        t, x, y, z = self.create_symbols("t x y z")
        invars = (x, y)
        if time:
            invars = (t,) + invars
        if dim == 3:
            invars += (z,)

        if isinstance(nu, str):
            nu = sp_parser.parse_expr(nu)
            if isinstance(nu, sp.Symbol):
                invars += (nu,)

        if isinstance(rho, str):
            rho = sp_parser.parse_expr(rho)
            if isinstance(rho, sp.Symbol):
                invars += (rho,)

        self.nu = nu
        self.rho = rho

        u = self.create_function("u", invars)
        v = self.create_function("v", invars)
        w = self.create_function("w", invars) if dim == 3 else sp.Number(0)
        p = self.create_function("p", invars)
        k = self.create_function("k", invars)
        epsilon = self.create_function("epsilon", invars)
        
        nu_t = C_mu * (self.k**2) / self.epsilon  
        P_k = nu_t * (u.diff(x)**2 + v.diff(y)**2 + w.diff(z)**2)

        continuity = u.diff(x) + v.diff(y) + w.diff(z)
        
        turbulence_energy_equation = (
            u * k.diff(x)
            + v * k.diff(y)
            + w * k.diff(z)
            - (
                (nu * k.diff(x)).diff(x)
                + (nu * k.diff(y)).diff(y)
                + (nu * k.diff(z)).diff(z)
            )
            + P_k  
            - epsilon  
        )

      
        turbulence_dissipation_equation = (
            u * epsilon.diff(x)
            + v * epsilon.diff(y)
            + w * epsilon.diff(z)
            - (
                (nu * epsilon.diff(x)).diff(x)
                + (nu * epsilon.diff(y)).diff(y)
                + (nu * epsilon.diff(z)).diff(z)
            )
            + self.C_epsilon_1 * (epsilon / k) * P_k  
            - self.C_epsilon_2 * epsilon**2  
        )
        
        reynolds_stress_x = -nu_t * (u.diff(x) + u.diff(y) + u.diff(z))  
        reynolds_stress_y = -nu_t * (v.diff(x) + v.diff(y) + v.diff(z))
        reynolds_stress_z = -nu_t * (w.diff(x) + w.diff(y) + w.diff(z))
        
        
        momentum_x = (
            u.diff(t)
            + u * u.diff(x)
            + v * u.diff(y)
            + w * u.diff(z)
            - (
                (nu * u.diff(x)).diff(x)
                + (nu * u.diff(y)).diff(y)
                + (nu * u.diff(z)).diff(z)
            )
            + 1 / rho * p.diff(x)
            + reynolds_stress_x
        )
        momentum_y = (
            v.diff(t)
            + u * v.diff(x)
            + v * v.diff(y)
            + w * v.diff(z)
            - (
                (nu * v.diff(x)).diff(x)
                + (nu * v.diff(y)).diff(y)
                + (nu * v.diff(z)).diff(z)
            )
            + 1 / rho * p.diff(y)
            + reynolds_stress_y
        )
        momentum_z = (
            w.diff(t)
            + u * w.diff(x)
            + v * w.diff(y)
            + w * w.diff(z)
            - (
                (nu * w.diff(x)).diff(x)
                + (nu * w.diff(y)).diff(y)
                + (nu * w.diff(z)).diff(z)
            )
            + 1 / rho * p.diff(z)
            + reynolds_stress_z
        )
        self.add_equation("continuity", continuity)
        self.add_equation("turbulence_energy_equation", turbulence_energy_equation)
        self.add_equation("turbulence_dissipation_equation", turbulence_dissipation_equation)
        self.add_equation("momentum_x", momentum_x)
        self.add_equation("momentum_y", momentum_y)
        if self.dim == 3:
            self.add_equation("momentum_z", momentum_z)

        self._apply_detach()
        
