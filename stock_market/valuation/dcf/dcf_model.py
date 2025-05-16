import numpy as np
from config import CONFIG

class DCFModel:
    def __init__(self, starting_fcf, wacc, high_growth_rate, terminal_growth_rate, net_debt, shares_outstanding,
                 high_growth_years, growth_decay, margin_compression, roic, wacc_drift_std, use_enhanced_model,
                 ticker, reinvestment_cap, scenario=None):
        self.starting_fcf = starting_fcf
        self.wacc = wacc
        self.high_growth_rate = high_growth_rate
        self.terminal_growth_rate = terminal_growth_rate
        self.net_debt = net_debt
        self.shares_outstanding = shares_outstanding
        self.high_growth_years = high_growth_years
        self.growth_decay = growth_decay
        self.margin_compression = margin_compression
        self.roic = roic
        self.wacc_drift_std = wacc_drift_std
        self.use_enhanced_model = use_enhanced_model
        self.ticker = ticker
        self.reinvestment_cap = reinvestment_cap
        self.scenario = scenario  # None for simple, 'bull' or 'doomsday' for enhanced

    def project_fcfs(self):
        fcfs = []
        fcf = self.starting_fcf
        wacc = self.wacc
        growth = self.high_growth_rate
        effective_growth_decay = self.growth_decay

        if self.use_enhanced_model:
            # Enhanced model with scenario-specific parameters
            if self.scenario == 'bull':
                growth *= 3.0  # Increase to 12% for bull case
                self.margin_compression = 0.0  # No compression for bull case
                self.wacc_drift_std *= 0.5  # Reduce to 0.001 for lower uncertainty
                wacc *= 0.625  # Reduce WACC to 5% for bull case
                effective_growth_decay = 0.005  # Slower decay for sustained growth
                self.reinvestment_cap = 0.0  # No reinvestment cap for bull
            elif self.scenario == 'doomsday':
                growth *= 0.125  # Reduce to 0.5% for doomsday scenario
                self.margin_compression *= 5.0  # Increase to 0.025 for stronger impact
                self.wacc_drift_std *= 2.5  # Increase to 0.005 for greater uncertainty
                effective_growth_decay = 0.025  # Faster decay for decline
                self.reinvestment_cap = 0.05  # Higher reinvestment cap for doomsday

            for t in range(self.high_growth_years):
                margin_comp = np.random.uniform(0, self.margin_compression)
                fcf *= (1 - margin_comp)
                if self.roic is not None and self.roic != 0:
                    reinvestment_rate = min((growth / self.roic) * (0.0 if self.scenario == 'bull' else 0.6), self.reinvestment_cap)
                    fcf *= (1 - reinvestment_rate)
                growth = max(growth - effective_growth_decay, 0.02)
                fcf *= (1 + growth)
                wacc += np.random.normal(0, self.wacc_drift_std)
                wacc = min(max(wacc, CONFIG['wacc_min']), CONFIG['wacc_max'])
                fcfs.append(fcf)
        else:
            # Simple model with logistic growth
            k = 0.1  # Growth rate decay constant
            carrying_capacity = 0.06  # Reduced to 6% for realism
            for t in range(self.high_growth_years):
                growth = carrying_capacity / (1 + (carrying_capacity / self.high_growth_rate - 1) * np.exp(-k * t))
                fcf *= (1 + growth)
                fcf *= np.random.normal(1, CONFIG['fcf_volatility'])
                fcfs.append(fcf)
        return fcfs

    def calculate_terminal_value(self, final_fcf):
        # Ensure terminal growth rate is less than WACC
        effective_terminal_rate = min(self.terminal_growth_rate, self.wacc - 0.01)
        if self.scenario == 'bull' and self.wacc > 0.06:
            effective_terminal_rate = min(0.04, self.wacc - 0.01)  # Allow up to 4% for bull case
        elif self.scenario == 'doomsday':
            effective_terminal_rate = min(0.015, self.wacc - 0.01)  # Cap at 1.5% for doomsday
        return final_fcf * (1 + effective_terminal_rate) / (self.wacc - effective_terminal_rate)

    def calculate_dcf(self):
        fcfs = self.project_fcfs()
        if not fcfs:
            return 0
        terminal_value = self.calculate_terminal_value(fcfs[-1])
        pv_fcfs = sum(fcf / (1 + self.wacc) ** (i + 1) for i, fcf in enumerate(fcfs))
        pv_terminal = terminal_value / (1 + self.wacc) ** self.high_growth_years
        enterprise_value = pv_fcfs + pv_terminal
        equity_value = max(enterprise_value - self.net_debt, 0)
        return equity_value / self.shares_outstanding if equity_value > 0 else 0