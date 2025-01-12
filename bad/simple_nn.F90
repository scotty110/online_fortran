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
        real(8), intent(in) :: x(:,:), x_prev(:,:), y_prev(:,:)
        ! Output values
        real(8), intent(out) :: y(:,:)

        ! Torch Types
        type(torch_tensor), dimension(3) :: in_tensors
        type(torch_tensor), dimension(1) :: out_tensors

        ! 2D layout
        integer :: tensor_layout_2d(3) = [2, 1]
        integer :: tensor_layout_3d(3) = [2, 1]  ! Specify layout for the output tensor

        ! Initialize the model if it has not been initialized yet
        if (.not. model_initialized) then
            call init_torch_model(model)
            model_initialized = .true.
        end if

        ! Make Torch Tensors for input
        call torch_tensor_from_array(in_tensors(1), x, tensor_layout_2d, torch_kCPU)
        call torch_tensor_from_array(in_tensors(2), x_prev, tensor_layout_2d, torch_kCPU)
        call torch_tensor_from_array(in_tensors(3), y_prev, tensor_layout_2d, torch_kCPU)

        ! Make Torch Tensor for output
        call torch_tensor_from_array(out_tensors(1), y, tensor_layout_3d, torch_kCPU)

        ! Perform inference
        call torch_model_forward(model, in_tensors, out_tensors)

        ! Extract the output tensor back into the output array `y`
        call torch_tensor_to_array(out_tensors(1), y)

        ! Free the torch tensors
        call torch_tensor_delete(in_tensors(1))
        call torch_tensor_delete(in_tensors(2))
        call torch_tensor_delete(in_tensors(3))
        call torch_tensor_delete(out_tensors(1))

    end subroutine torch_inference

 end module online_nn