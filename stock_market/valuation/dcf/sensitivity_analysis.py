from config import CONFIG, logger
from dcf_model import DCFModel

def perform_sensitivity_analysis(inputs, mean_val, model_type, ticker):
    logger.info(f"\nSensitivity Analysis for {model_type} Model:")
    scenarios = ['optimal', 'doomsday'] if model_type == 'Enhanced' else [None]
    
    for scenario in scenarios:
        if model_type == 'Enhanced':
            logger.info(f"\nScenario: {scenario.capitalize()}")
        for param, change in [('WACC', 0.01), ('High Growth Rate', 0.01), ('Terminal Growth Rate', 0.005)]:
            temp_model = DCFModel(
                starting_fcf=inputs['starting_fcf'],
                wacc=inputs['wacc'] + change if param == 'WACC' else inputs['wacc'],
                high_growth_rate=inputs['high_growth_rate'] + change if param == 'High Growth Rate' else inputs['high_growth_rate'],
                terminal_growth_rate=inputs['terminal_growth_rate'] + change if param == 'Terminal Growth Rate' else inputs['terminal_growth_rate'],
                net_debt=inputs['net_debt'],
                shares_outstanding=inputs['shares_outstanding'],
                high_growth_years=CONFIG['high_growth_years'],
                growth_decay=inputs.get('growth_decay', 0),
                margin_compression=inputs.get('margin_compression', 0),
                roic=inputs['roic'],
                wacc_drift_std=inputs.get('wacc_drift_std', 0),
                use_enhanced_model=model_type == 'Enhanced',
                ticker=ticker,
                reinvestment_cap=inputs['reinvestment_cap'],
                scenario=scenario
            )
            new_value = temp_model.calculate_dcf()
            logger.info(f"{param} +{change*100:.1f}%: Fair Value = {new_value:.2f} ({(new_value - mean_val)/mean_val*100:.1f}% change)")