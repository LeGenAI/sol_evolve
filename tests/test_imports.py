def test_import_package():
    import solevolve

    assert hasattr(solevolve, "build_sol_evolve_network")
