!----------------------------------------------------------------------------------------
!
! DGF (Deep Geometry Framework) model for H-EFTCAMB
!
! Horndeski nKGB with:
!   G2 = X - V(phi),  V(phi) = F0*(phi - ln(phi)),  F0 = 5.518
!   G3 = 9.708 * X    (= 6*phi_gr * X, FIXED constant)
!   G4 = M_Pl^2/2
!   G5 = 0
!
! Zero free gravity parameters. phi_gr = golden ratio = 1.618033988749895
!
! Based on 008p6_Scaling_Cubic.f90 (Albuquerque & Frusciante).
! Adapted for DGF logarithmic potential. J. Shields, 2026.
!
! STATUS: Scaffold with potential functions and EFT mapping stubs.
! The background ODE solver requires careful derivation of the
! Friedmann + Klein-Gordon system with the logarithmic potential.
! For production use, hi_class with tabulated_alphas remains the
! recommended backend until this module is completed.
!
!----------------------------------------------------------------------------------------

module EFTCAMB_full_DGF

    use precision
    use IniObjects
    use MpiUtils
    use FileUtils
    use EFTCAMB_cache
    use EFT_def
    use EFTCAMB_abstract_model_full
    use equispaced_linear_interpolation_1D

    implicit none
    private
    public EFTCAMB_DGF

    ! DGF physical constants (all derived from c = phi_gr)
    real(dl), parameter :: PHI_GR  = 1.618033988749895_dl
    real(dl), parameter :: F0_DGF  = 5.518_dl               ! 3*phi*(phi - ln(phi))
    real(dl), parameter :: A_DGF   = 9.708_dl               ! 6*phi_gr (G3 coupling)

    type, extends ( EFTCAMB_full_model ) :: EFTCAMB_DGF

        type(equispaced_linear_interpolate_function_1D) :: fun_x1
        type(equispaced_linear_interpolate_function_1D) :: fun_y1
        type(equispaced_linear_interpolate_function_1D) :: fun_phi

        integer  :: num_points = 10000
        real(dl) :: x_initial  = log(1._dl/(1.5d0*1.d5 + 1._dl))
        real(dl) :: x_final    = 0._dl
        logical  :: debug_flag = .false.

    contains
        procedure :: read_model_selection            => DGFReadSel
        procedure :: allocate_model_selection        => DGFAllocSel
        procedure :: init_model_parameters           => DGFInitParams
        procedure :: init_model_parameters_from_file => DGFInitParamsFile
        procedure :: initialize_background           => DGFInitBG
        procedure :: solve_background                => DGFSolveBG
        procedure :: compute_param_number            => DGFParamNum
        procedure :: feedback                        => DGFFeedback
        procedure :: parameter_names                 => DGFParamNames
        procedure :: parameter_names_latex           => DGFParamNamesLatex
        procedure :: parameter_values                => DGFParamValues
        procedure :: compute_background_EFT_functions  => DGFBgEFT
        procedure :: compute_secondorder_EFT_functions => DGF2ndEFT
        procedure :: compute_adotoa                    => DGFAdotoa
        procedure :: compute_H_derivs                  => DGFHDeriv
        procedure :: additional_model_stability        => DGFStability
    end type EFTCAMB_DGF

contains

    ! ── DGF potential functions (pure, usable anywhere) ───────────────────

    pure function V_dgf(phi) result(V)
        real(dl), intent(in) :: phi
        real(dl) :: V
        V = F0_DGF * (phi - log(phi))    ! V(phi) = F0*(phi - ln(phi))
    end function

    pure function Vp_dgf(phi) result(Vp)
        real(dl), intent(in) :: phi
        real(dl) :: Vp
        Vp = F0_DGF * (1._dl - 1._dl/phi)   ! V'(phi) = F0*(1 - 1/phi)
    end function

    pure function Vpp_dgf(phi) result(Vpp)
        real(dl), intent(in) :: phi
        real(dl) :: Vpp
        Vpp = F0_DGF / phi**2                ! V''(phi) = F0/phi^2
    end function

    ! effective lambda(phi) = -V'/V (sign convention matching SCG)
    pure function lambda_dgf(phi) result(lam)
        real(dl), intent(in) :: phi
        real(dl) :: lam
        lam = -(1._dl - 1._dl/phi) / (phi - log(phi))
    end function

    ! Gamma(phi) = V*V''/(V')^2 = (phi - ln(phi)) / (phi - 1)^2
    pure function Gamma_dgf(phi) result(Gam)
        real(dl), intent(in) :: phi
        real(dl) :: Gam
        if (abs(phi - 1._dl) < 1.d-10) then
            Gam = 1._dl   ! limit as phi -> 1
        else
            Gam = (phi - log(phi)) / (phi - 1._dl)**2
        end if
    end function

    ! ── Required interface stubs (zero free parameters) ───────────────────

    subroutine DGFReadSel(self, Ini, eft_error)
        class(EFTCAMB_DGF) :: self; type(TIniFile) :: Ini; integer :: eft_error
        self%debug_flag = Ini%Read_logical('DGF_debug', .false.)
    end subroutine

    subroutine DGFAllocSel(self, Ini, eft_error)
        class(EFTCAMB_DGF) :: self; type(TIniFile) :: Ini; integer :: eft_error
    end subroutine

    subroutine DGFInitParams(self, array)
        class(EFTCAMB_DGF) :: self
        real(dl), dimension(self%parameter_number), intent(in) :: array
    end subroutine

    subroutine DGFInitParamsFile(self, Ini, eft_error)
        class(EFTCAMB_DGF) :: self; type(TIniFile) :: Ini; integer :: eft_error
    end subroutine

    subroutine DGFParamNum(self)
        class(EFTCAMB_DGF) :: self
        self%parameter_number = 0
    end subroutine

    subroutine DGFFeedback(self, print_params)
        class(EFTCAMB_DGF) :: self; logical, optional :: print_params
        write(*,'(a)')       ' ========================================'
        write(*,'(a)')       '  Deep Geometry Framework (DGF)'
        write(*,'(a)')       '  G3 = 9.708*X | V = 5.518*(phi-ln(phi))'
        write(*,'(a)')       '  Free gravity parameters: 0'
        write(*,'(a)')       ' ========================================'
    end subroutine

    subroutine DGFParamNames(self, i, name)
        class(EFTCAMB_DGF) :: self; integer, intent(in) :: i; character(*), intent(out) :: name
        name = ''
    end subroutine

    subroutine DGFParamNamesLatex(self, i, name)
        class(EFTCAMB_DGF) :: self; integer, intent(in) :: i; character(*), intent(out) :: name
        name = ''
    end subroutine

    subroutine DGFParamValues(self, i, value)
        class(EFTCAMB_DGF) :: self; integer, intent(in) :: i; real(dl), intent(out) :: value
        value = 0._dl
    end subroutine

    ! ── Background solver ─────────────────────────────────────────────────
    ! TODO: Implement the full background ODE for DGF.
    ! The system is 3 equations (x1, y1, phi) vs SCG's 3 (x1, y1, y2).
    ! Key difference: phi must be tracked because lambda(phi) is not constant.
    ! The Friedmann constraint and acceleration equation need the G3 = A*X
    ! contribution with the DGF-specific potential slope.
    !
    ! For now these return failure, and hi_class should be used instead.

    subroutine DGFInitBG(self, params_cache, feedback_level, success)
        class(EFTCAMB_DGF) :: self
        type(TEFTCAMB_parameter_cache), intent(in) :: params_cache
        integer, intent(in) :: feedback_level
        logical, intent(out) :: success
        success = .false.
        if (feedback_level > 0) write(*,'(a)') ' DGF: BG solver stub — use hi_class'
    end subroutine

    subroutine DGFSolveBG(self, params_cache, feedback_level, success)
        class(EFTCAMB_DGF) :: self
        type(TEFTCAMB_parameter_cache), intent(in) :: params_cache
        integer, intent(in) :: feedback_level
        logical, intent(out) :: success
        success = .false.
    end subroutine

    ! ── EFT function mapping stubs ────────────────────────────────────────
    ! TODO: Once background is solved, fill these from the interpolated
    ! x1(a), y1(a), phi(a) solutions. The DGF EFT functions are:
    !   alpha_M = 0, alpha_T = 0 (G4 = const, G5 = 0)
    !   alpha_B = -2*A*x1*H  (braiding from G3)
    !   alpha_K = 2*x1^2 + 12*A^2*x1^4*H^2  (kinetic from G2+G3)

    subroutine DGFBgEFT(self, a, pc, tc)
        class(EFTCAMB_DGF) :: self; real(dl), intent(in) :: a
        type(TEFTCAMB_parameter_cache), intent(inout) :: pc
        type(TEFTCAMB_timestep_cache),  intent(inout) :: tc
        tc%EFTOmegaV = 0._dl; tc%EFTOmegaP = 0._dl
        tc%EFTOmegaPP = 0._dl; tc%EFTOmegaPPP = 0._dl
        tc%EFTc = 0._dl; tc%EFTcdot = 0._dl
        tc%EFTLambda = 0._dl; tc%EFTLambdadot = 0._dl
    end subroutine

    subroutine DGF2ndEFT(self, a, pc, tc)
        class(EFTCAMB_DGF) :: self; real(dl), intent(in) :: a
        type(TEFTCAMB_parameter_cache), intent(inout) :: pc
        type(TEFTCAMB_timestep_cache),  intent(inout) :: tc
        tc%EFTGamma1V = 0._dl; tc%EFTGamma1P = 0._dl
        tc%EFTGamma2V = 0._dl; tc%EFTGamma2P = 0._dl
        tc%EFTGamma3V = 0._dl; tc%EFTGamma3P = 0._dl
        tc%EFTGamma3PP = 0._dl; tc%EFTGamma3PPP = 0._dl; tc%EFTGamma3PPPP = 0._dl
        tc%EFTGamma4V = 0._dl; tc%EFTGamma4P = 0._dl; tc%EFTGamma4PP = 0._dl
        tc%EFTGamma5V = 0._dl; tc%EFTGamma5P = 0._dl
        tc%EFTGamma6V = 0._dl; tc%EFTGamma6P = 0._dl
    end subroutine

    subroutine DGFAdotoa(self, a, pc, tc)
        class(EFTCAMB_DGF) :: self; real(dl), intent(in) :: a
        type(TEFTCAMB_parameter_cache), intent(inout) :: pc
        type(TEFTCAMB_timestep_cache),  intent(inout) :: tc
        tc%adotoa = 0._dl
    end subroutine

    subroutine DGFHDeriv(self, a, pc, tc)
        class(EFTCAMB_DGF) :: self; real(dl), intent(in) :: a
        type(TEFTCAMB_parameter_cache), intent(inout) :: pc
        type(TEFTCAMB_timestep_cache),  intent(inout) :: tc
        tc%Hdot = 0._dl; tc%Hdotdot = 0._dl; tc%Hdotdotdot = 0._dl
    end subroutine

    function DGFStability(self, a, pc, tc)
        class(EFTCAMB_DGF) :: self; real(dl), intent(in) :: a
        type(TEFTCAMB_parameter_cache), intent(inout) :: pc
        type(TEFTCAMB_timestep_cache),  intent(inout) :: tc
        logical :: DGFStability
        DGFStability = .True.
    end function

end module EFTCAMB_full_DGF
