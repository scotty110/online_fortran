module online_nn
    use ftorch
    implicit none
 
    public :: torch_inference
 
    ! Declare the torch model without initializing it here
    type(torch_model), save :: model
    logical, save :: model_initialized = .false.
 
 contains
 
    subroutine init_torch_model(model)
       ! Initialize the model
       type(torch_model), intent(inout) :: model
       call torch_model_load(model, "./online_example.pt", torch_kCPU)
    end subroutine init_torch_model
 
    subroutine torch_inference(x, x_prev, y_prev, y)
        ! Input values
        real(8), intent(in) :: x(:), x_prev(:), y_prev(:)
        ! Output values
        real(8), intent(out) :: y(:)

        ! Torch Types
        type(torch_tensor), dimension(3) :: in_tensors
        type(torch_tensor), dimension(1) :: out_tensors

        ! Scalar layout (no dimensionality for scalars)
        integer :: scalar_layout(1) = [1]

        ! Initialize the model if it has not been initialized yet
        if (.not. model_initialized) then
            call init_torch_model(model)
            model_initialized = .true.
        end if

        ! Make Torch Tensors for scalar input
        call torch_tensor_from_array(in_tensors(1), x, torch_kCPU)
        call torch_tensor_from_array(in_tensors(2), x_prev, torch_kCPU)
        call torch_tensor_from_array(in_tensors(3), y_prev, torch_kCPU)

        ! Make Torch Tensor for scalar output
        call torch_tensor_from_scalar(out_tensors(1), y, torch_kCPU)

        ! Perform inference
        call torch_model_forward(model, in_tensors, out_tensors)

        ! Extract the output scalar back into the variable `y`
        call torch_tensor_to_scalar(out_tensors(1), y)

        ! Free the torch tensors
        call torch_tensor_delete(in_tensors(1))
        call torch_tensor_delete(in_tensors(2))
        call torch_tensor_delete(in_tensors(3))
        call torch_tensor_delete(out_tensors(1))

    end subroutine torch_inference

 end module online_nn