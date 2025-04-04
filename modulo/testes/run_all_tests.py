#!/usr/bin/env python3
"""
Executor de Testes do Sistema de Consulta em Linguagem Natural
=============================================================

Este script executa todos os testes do sistema, incluindo:
- Testes unitários
- Testes de integração
- Cenários de testes
- Benchmarks de desempenho

Pode ser executado com diferentes opções para focar em tipos específicos de testes.
"""

import os
import sys
import unittest
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# Adiciona o diretório pai ao PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa cenários de teste
from testes.test_scenarios import (
    scenario1_basic_queries,
    scenario2_data_analysis,
    scenario3_error_handling
)


def run_unit_tests(verbose: bool = False) -> Dict[str, Any]:
    """
    Executa todos os testes unitários
    
    Args:
        verbose: Se True, exibe detalhes dos testes
    
    Returns:
        Dict com os resultados dos testes
    """
    start_time = time.time()
    print("\n" + "="*50)
    print("Executando testes unitários...")
    print("="*50)
    
    # Descobre e carrega todos os testes unitários 
    # (arquivos que começam com 'test_' e não estão na pasta 'test_scenarios')
    test_suite = unittest.defaultTestLoader.discover(
        os.path.dirname(os.path.abspath(__file__)),
        pattern='test_*.py',
        top_level_dir=os.path.dirname(os.path.abspath(__file__))
    )
    
    # Filtra para excluir os arquivos de cenários
    filtered_suite = unittest.TestSuite()
    for suite in test_suite:
        for test_case in suite:
            if 'test_scenarios' not in str(test_case):
                filtered_suite.addTest(test_case)
    
    # Executa os testes
    test_runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    test_result = test_runner.run(filtered_suite)
    
    # Calcula estatísticas
    duration = time.time() - start_time
    total_tests = test_result.testsRun
    failures = len(test_result.failures)
    errors = len(test_result.errors)
    skipped = len(test_result.skipped) if hasattr(test_result, 'skipped') else 0
    success_count = total_tests - failures - errors - skipped
    success_rate = (success_count / total_tests) * 100 if total_tests > 0 else 0
    
    # Retorna os resultados
    results = {
        "type": "unit_tests",
        "total_tests": total_tests,
        "passed": success_count,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "success_rate": success_rate,
        "duration_seconds": duration,
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"\nTestes unitários concluídos em {duration:.2f} segundos")
    print(f"Total: {total_tests}, Sucesso: {success_count}, Falhas: {failures}, Erros: {errors}, Pulados: {skipped}")
    print(f"Taxa de sucesso: {success_rate:.1f}%")
    
    return results


def run_scenarios(scenarios: List[str] = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Executa cenários de teste específicos ou todos
    
    Args:
        scenarios: Lista de cenários para executar, ou None para executar todos
        verbose: Se True, exibe detalhes adicionais
    
    Returns:
        Dict com os resultados dos cenários
    """
    start_time = time.time()
    print("\n" + "="*50)
    print("Executando cenários de teste...")
    print("="*50)
    
    # Define os cenários disponíveis
    available_scenarios = {
        "scenario1": scenario1_basic_queries.execute_scenario,
        "scenario2": scenario2_data_analysis.execute_scenario,
        "scenario3": scenario3_error_handling.execute_scenario
    }
    
    # Determina quais cenários executar
    if scenarios:
        scenarios_to_run = {name: func for name, func in available_scenarios.items() if name in scenarios}
    else:
        scenarios_to_run = available_scenarios
    
    if not scenarios_to_run:
        print("Nenhum cenário válido especificado.")
        return {
            "type": "scenarios",
            "error": "No valid scenarios specified",
            "timestamp": datetime.now().isoformat()
        }
    
    # Executa cada cenário
    scenario_results = {}
    for name, execute_func in scenarios_to_run.items():
        scenario_start_time = time.time()
        print(f"\nExecutando cenário: {name}")
        
        try:
            # Executa o cenário
            results = execute_func()
            
            # Calcula estatísticas
            success_count = sum(1 for r in results if r.get('success', False) or r.get('outcome') == 'success')
            total_tests = len(results)
            success_rate = (success_count / total_tests) * 100 if total_tests > 0 else 0
            
            scenario_results[name] = {
                "total_tests": total_tests,
                "success_count": success_count,
                "failure_count": total_tests - success_count,
                "success_rate": success_rate,
                "duration_seconds": time.time() - scenario_start_time
            }
            
            print(f"Cenário {name} concluído: {success_count}/{total_tests} testes bem-sucedidos ({success_rate:.1f}%)")
            
        except Exception as e:
            print(f"Erro ao executar cenário {name}: {str(e)}")
            scenario_results[name] = {
                "error": str(e),
                "duration_seconds": time.time() - scenario_start_time
            }
    
    # Calcula estatísticas gerais
    duration = time.time() - start_time
    
    # Retorna os resultados
    results = {
        "type": "scenarios",
        "scenarios": scenario_results,
        "total_scenarios": len(scenario_results),
        "duration_seconds": duration,
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"\nTodos os cenários de teste concluídos em {duration:.2f} segundos")
    
    return results


def run_all_tests(args) -> None:
    """
    Executa todos os testes, de acordo com os argumentos fornecidos
    
    Args:
        args: Argumentos de linha de comando
    """
    os.makedirs("testes/output", exist_ok=True)
    
    all_results = {}
    start_time = time.time()
    
    # Executa testes unitários, se solicitado
    if args.unit or args.all:
        unit_test_results = run_unit_tests(args.verbose)
        all_results["unit_tests"] = unit_test_results
    
    # Executa cenários de teste, se solicitado
    if args.scenarios or args.all:
        scenario_results = run_scenarios(args.scenario, args.verbose)
        all_results["scenarios"] = scenario_results
    
    # Calcula duração total
    all_results["total_duration_seconds"] = time.time() - start_time
    all_results["execution_date"] = datetime.now().isoformat()
    
    # Salva os resultados em um arquivo JSON
    results_file = os.path.join("testes/output", "test_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*50)
    print(f"Execução de testes concluída em {all_results['total_duration_seconds']:.2f} segundos")
    print(f"Resultados salvos em: {results_file}")
    print("="*50)


if __name__ == "__main__":
    # Configura o parser de argumentos
    parser = argparse.ArgumentParser(description="Executor de testes do Sistema de Consulta em Linguagem Natural")
    
    # Grupos de testes
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Executa todos os testes")
    group.add_argument("--unit", action="store_true", help="Executa apenas testes unitários")
    group.add_argument("--scenarios", action="store_true", help="Executa apenas cenários de teste")
    
    # Seleção de cenários
    parser.add_argument("--scenario", nargs="+", choices=["scenario1", "scenario2", "scenario3"], 
                      help="Especifica quais cenários executar")
    
    # Outras opções
    parser.add_argument("--verbose", "-v", action="store_true", help="Exibe detalhes adicionais durante os testes")
    
    # Parse os argumentos
    args = parser.parse_args()
    
    # Se nenhum grupo foi especificado, assume --all
    if not (args.all or args.unit or args.scenarios):
        args.all = True
    
    # Executa os testes
    run_all_tests(args)